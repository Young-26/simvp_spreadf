#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict / case-study script for simvp_spreadf.

Features
--------
1) Load best.ckpt (or any ckpt) and run per-sample evaluation on the validation set.
2) Save metrics to JSONL, one line per sample.
3) Auto-select sample groups: worst / best / typical / pred2_worse.
4) Re-run only selected samples and optionally save:
   - mosaic images (8 inputs + 2 GT + 2 Pred)
   - standalone pred_01.png / pred_02.png
5) Optionally export pred_01.png / pred_02.png for ALL validation samples,
   with one subdirectory per sample.

Designed to match repo structure at commit:
971dfdf05dee5a4120a5e0ea73b4f89e8ed302ef
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.ionogram_manifest import IonogramManifestDataset
from simvp.wrapper import SimVPForecast


# -----------------------------
# Utilities
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sample evaluation and case-study export for simvp_spreadf")
    parser.add_argument("--val_manifest", type=str, required=True, help="Validation manifest JSONL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. best.ckpt")

    parser.add_argument("--output_dir", type=str, default="./predict_outputs", help="Root output directory")
    parser.add_argument("--metrics_path", type=str, default=None, help="Optional explicit path for per_sample_metrics.jsonl")
    parser.add_argument("--selection_json_path", type=str, default=None, help="Optional explicit path for auto_selected_samples.json")
    parser.add_argument("--selected_rows_path", type=str, default=None, help="Optional explicit path for selected_rows__*.json")
    parser.add_argument("--mosaic_dir", type=str, default=None, help="Optional explicit mosaic output directory")
    parser.add_argument("--pred_frames_dir", type=str, default=None, help="Optional explicit pred frame output directory for selected rows")
    parser.add_argument("--all_pred_frames_dir", type=str, default=None, help="Optional explicit pred frame output directory for ALL rows")

    parser.add_argument("--batch_size", type=int, default=None, help="Inference batch size. Defaults to ckpt val_batch_size/batch_size/4")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers. Defaults to ckpt num_workers/4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use autocast during inference")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory")

    parser.add_argument("--save_selected", action="store_true", help="Save mosaics for selected groups")
    parser.add_argument("--save_pred_frames", action="store_true", help="Save standalone pred_01.png / pred_02.png for selected groups")
    parser.add_argument("--save_all_pred_frames", action="store_true", help="Save standalone pred_01.png / pred_02.png for ALL validation samples")
    parser.add_argument("--selected_json", type=str, default=None, help="Optional custom selected json, supports dataset_indices/sample_ids")
    parser.add_argument("--selected_name", type=str, default=None, help="Name suffix for selected output group")

    parser.add_argument("--worst_k", type=int, default=20)
    parser.add_argument("--best_k", type=int, default=10)
    parser.add_argument("--typical_k", type=int, default=10)
    parser.add_argument("--pred2_worse_k", type=int, default=20)

    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick profiling / debug")
    parser.add_argument("--skip_ssim", action="store_true", help="Skip SSIM to speed up evaluation")

    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint format not supported: {type(ckpt)}")
    return ckpt


def coalesce(*values, default=None):
    for v in values:
        if v is not None:
            return v
    return default


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def jsonl_dump(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sanitize_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    return name.strip("_") or "sample"


# -----------------------------
# Dataset / loader helpers
# -----------------------------

def collate_with_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    x_local = None
    y_local = None
    if "x_local" in batch[0]:
        x_local = torch.stack([item["x_local"] for item in batch], dim=0)
    if "y_local" in batch[0]:
        y_local = torch.stack([item["y_local"] for item in batch], dim=0)
    dataset_indices = [int(item.get("dataset_idx", -1)) for item in batch]
    sample_ids = [item.get("sample_id", str(i)) for i, item in enumerate(batch)]
    timestamps = [item.get("timestamps", None) for item in batch]
    sequence_ids = [item.get("sequence_id", None) for item in batch]
    out = {
        "x": x,
        "y": y,
        "dataset_indices": dataset_indices,
        "sample_ids": sample_ids,
        "timestamps": timestamps,
        "sequence_ids": sequence_ids,
    }
    if x_local is not None:
        out["x_local"] = x_local
    if y_local is not None:
        out["y_local"] = y_local
    return out


def attach_sample_metadata(dataset: IonogramManifestDataset) -> None:
    for idx, item in enumerate(dataset.samples):
        item["dataset_idx"] = idx
        item["sample_id"] = resolve_sample_id(item, idx)


def resolve_sample_id(item: Dict[str, Any], dataset_idx: int) -> str:
    timestamps = item.get("timestamps", None)
    if isinstance(timestamps, list) and len(timestamps) > 0 and timestamps[0] not in (None, ""):
        return str(timestamps[0])
    if item.get("sequence_id", None) not in (None, ""):
        return str(item["sequence_id"])
    return str(dataset_idx)


def validate_dataset_sequence_lengths(dataset: IonogramManifestDataset, in_T: int, out_T: int) -> None:
    if len(dataset) == 0:
        return

    sample = dataset[0]
    sample_in_T = int(sample["x"].shape[0])
    sample_out_T = int(sample["y"].shape[0])
    if sample_in_T != in_T or sample_out_T != out_T:
        raise ValueError(
            f"Dataset provides input/target lengths ({sample_in_T}, {sample_out_T}), but "
            f"the checkpoint expects ({in_T}, {out_T})."
        )


# -----------------------------
# Metrics
# -----------------------------

def mae_per_frame(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().flatten(2).mean(dim=2)


def mse_per_frame(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).flatten(2).mean(dim=2)


def psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    return 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse, min=eps))


def ssim_per_frame(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    bsz, t_len, c, _, _ = pred_np.shape
    out = np.zeros((bsz, t_len), dtype=np.float32)

    for b in range(bsz):
        for t in range(t_len):
            if c == 1:
                p = pred_np[b, t, 0]
                g = target_np[b, t, 0]
                score = skimage_ssim(p, g, data_range=data_range)
            else:
                p = np.transpose(pred_np[b, t], (1, 2, 0))
                g = np.transpose(target_np[b, t], (1, 2, 0))
                score = skimage_ssim(p, g, data_range=data_range, channel_axis=2)
            out[b, t] = float(score)
    return torch.from_numpy(out)


# -----------------------------
# Image export helpers
# -----------------------------

def chw_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    arr = img_t.detach().cpu().float().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def uint8_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


def pad_with_caption(img: Image.Image, caption: str, font_pad: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + font_pad), color=(255, 255, 255))
    canvas.paste(img.convert("RGB"), (0, font_pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), caption, fill=(0, 0, 0))
    return canvas


def build_mosaic(
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    sample_id: str,
    dataset_idx: int,
) -> Image.Image:
    rows: List[List[Image.Image]] = []

    def make_row(tensors: torch.Tensor, prefix: str) -> List[Image.Image]:
        row = []
        for i in range(tensors.shape[0]):
            pil = uint8_to_pil(chw_to_uint8(tensors[i])).convert("RGB")
            row.append(pad_with_caption(pil, f"{prefix}{i+1}"))
        return row

    rows.append(make_row(x, "In"))
    rows.append(make_row(y, "GT"))
    rows.append(make_row(pred, "Pred"))

    max_cols = max(len(r) for r in rows)
    tile_w = max(img.width for r in rows for img in r)
    tile_h = max(img.height for r in rows for img in r)
    gap = 8
    header_h = 34

    mosaic_w = max_cols * tile_w + (max_cols - 1) * gap
    mosaic_h = header_h + len(rows) * tile_h + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (mosaic_w, mosaic_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), f"idx={dataset_idx} | sample_id={sample_id}", fill=(0, 0, 0))

    y0 = header_h
    for row in rows:
        x0 = 0
        for img in row:
            canvas.paste(img, (x0, y0))
            x0 += tile_w + gap
        y0 += tile_h + gap

    return canvas


def save_pred_frames(sample_dir: Path, pred: torch.Tensor) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i in range(pred.shape[0]):
        pil = uint8_to_pil(chw_to_uint8(pred[i]))
        pil.save(sample_dir / f"pred_{i+1:02d}.png")


# -----------------------------
# Selection helpers
# -----------------------------

def safe_float(v: Any) -> float:
    return float(v) if v is not None else float("nan")


def row_distance_to_center(row: Dict[str, Any], mean_vals: Dict[str, float], std_vals: Dict[str, float]) -> float:
    keys = ["mae_avg", "ssim_avg", "psnr_avg"]
    s = 0.0
    for k in keys:
        x = safe_float(row[k])
        mu = mean_vals[k]
        sd = std_vals[k] if std_vals[k] > 0 else 1.0
        s += ((x - mu) / sd) ** 2
    return math.sqrt(s)


def unique_by_dataset_idx(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        idx = int(row["dataset_idx"])
        if idx in seen:
            continue
        seen.add(idx)
        out.append(row)
    return out


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep_keys = [
        "sample_id",
        "dataset_idx",
        "mae_pred1",
        "mae_pred2",
        "mae_avg",
        "ssim_pred1",
        "ssim_pred2",
        "ssim_avg",
        "psnr_pred1",
        "psnr_pred2",
        "psnr_avg",
    ]
    return [{k: to_builtin(r[k]) for k in keep_keys if k in r} for r in rows]


def auto_select(rows: List[Dict[str, Any]], worst_k: int, best_k: int, typical_k: int, pred2_worse_k: int) -> Dict[str, Any]:
    arr = {
        "mae_avg": np.array([safe_float(r["mae_avg"]) for r in rows], dtype=np.float64),
        "ssim_avg": np.array([safe_float(r["ssim_avg"]) for r in rows], dtype=np.float64),
        "psnr_avg": np.array([safe_float(r["psnr_avg"]) for r in rows], dtype=np.float64),
    }
    mean_vals = {k: float(np.nanmean(v)) for k, v in arr.items()}
    std_vals = {k: float(np.nanstd(v)) for k, v in arr.items()}

    worst_by_ssim = unique_by_dataset_idx(sorted(rows, key=lambda r: (safe_float(r["ssim_avg"]), -safe_float(r["mae_avg"])))[:worst_k])
    worst_by_mae = unique_by_dataset_idx(sorted(rows, key=lambda r: (-safe_float(r["mae_avg"]), safe_float(r["ssim_avg"])))[:worst_k])
    best_by_ssim = unique_by_dataset_idx(sorted(rows, key=lambda r: (-safe_float(r["ssim_avg"]), safe_float(r["mae_avg"])))[:best_k])
    best_by_mae = unique_by_dataset_idx(sorted(rows, key=lambda r: (safe_float(r["mae_avg"]), -safe_float(r["ssim_avg"])))[:best_k])

    typical_sorted = sorted(rows, key=lambda r: row_distance_to_center(r, mean_vals, std_vals))
    typical_by_center = unique_by_dataset_idx(typical_sorted[:typical_k])

    def pred2_gap_score(r: Dict[str, Any]) -> Tuple[float, float]:
        ssim_gap = safe_float(r["ssim_pred1"]) - safe_float(r["ssim_pred2"])
        mae_gap = safe_float(r["mae_pred2"]) - safe_float(r["mae_pred1"])
        return (ssim_gap + mae_gap, ssim_gap)

    pred2_worse = unique_by_dataset_idx(sorted(rows, key=pred2_gap_score, reverse=True)[:pred2_worse_k])

    gallery_default = {
        "worst_20": summarize_rows(worst_by_ssim[:worst_k]),
        "best_10": summarize_rows(best_by_ssim[:best_k]),
        "typical_10": summarize_rows(typical_by_center[:typical_k]),
    }

    return {
        "summary": {
            "count": len(rows),
            "mean": mean_vals,
            "std": std_vals,
        },
        "worst_by_ssim_avg": summarize_rows(worst_by_ssim),
        "worst_by_mae_avg": summarize_rows(worst_by_mae),
        "best_by_ssim_avg": summarize_rows(best_by_ssim),
        "best_by_mae_avg": summarize_rows(best_by_mae),
        "typical_by_center": summarize_rows(typical_by_center),
        "pred2_worse": summarize_rows(pred2_worse),
        "gallery_default": gallery_default,
    }


def collect_selected_indices(selected_obj: Dict[str, Any]) -> List[int]:
    out: List[int] = []

    def visit(x: Any):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "dataset_indices" and isinstance(v, list):
                    for t in v:
                        out.append(int(t))
                else:
                    visit(v)
        elif isinstance(x, list):
            for item in x:
                if isinstance(item, dict) and "dataset_idx" in item:
                    out.append(int(item["dataset_idx"]))
                else:
                    visit(item)

    visit(selected_obj)
    out = sorted(set(out))
    return out


def load_custom_selected(path: str, all_rows: List[Dict[str, Any]], selected_name: Optional[str]) -> Tuple[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    name = selected_name or f"custom_from_{sanitize_name(Path(path).stem)}"
    by_idx = {int(r["dataset_idx"]): r for r in all_rows}
    by_sid = {str(r["sample_id"]): r for r in all_rows}

    rows: List[Dict[str, Any]] = []
    idxs = obj.get("dataset_indices", []) if isinstance(obj, dict) else []
    sids = obj.get("sample_ids", []) if isinstance(obj, dict) else []

    for idx in idxs:
        idx = int(idx)
        if idx in by_idx:
            rows.append(by_idx[idx])
    for sid in sids:
        sid = str(sid)
        if sid in by_sid:
            rows.append(by_sid[sid])

    if not rows:
        nested_idxs = collect_selected_indices(obj)
        for idx in nested_idxs:
            if idx in by_idx:
                rows.append(by_idx[idx])

    rows = unique_by_dataset_idx(rows)
    return name, rows


def load_gallery_default(auto_selected_obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    rows = []
    for key in ["worst_20", "best_10", "typical_10"]:
        rows.extend(auto_selected_obj.get("gallery_default", {}).get(key, []))
    rows = unique_by_dataset_idx(rows)
    return "gallery_default", rows


# -----------------------------
# Evaluation / export
# -----------------------------

def extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    model_state = ckpt.get("model")
    if isinstance(model_state, dict):
        return model_state
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError("Checkpoint does not contain a model state dict under 'model'.")


def build_model_from_ckpt_args(
    ckpt_args: Dict[str, Any],
    image_mode: str,
    image_size: int,
) -> Tuple[SimVPForecast, Dict[str, Any]]:
    channels = 1 if image_mode == "L" else 3
    in_T = int(ckpt_args.get("in_T", 8))
    out_T = int(ckpt_args.get("out_T", 2))
    arch = str(ckpt_args.get("arch", "simvp"))
    simvp_model_type = str(ckpt_args.get("simvp_model_type", "incepu"))
    predrnnpp_recipe = str(ckpt_args.get("predrnnpp_recipe", "simvp"))
    use_local_branch = bool(ckpt_args.get("use_local_branch", False))
    local_top = int(ckpt_args.get("local_top", 186))
    local_bottom = int(ckpt_args.get("local_bottom", 410))
    local_crop = (local_top, local_bottom)

    model = SimVPForecast(
        in_T=in_T,
        out_T=out_T,
        C=channels,
        H=image_size,
        W=image_size,
        hid_S=int(ckpt_args.get("hid_S", 32)),
        hid_T=int(ckpt_args.get("hid_T", 128)),
        N_S=int(ckpt_args.get("N_S", 4)),
        N_T=int(ckpt_args.get("N_T", 4)),
        simvp_model_type=simvp_model_type,
        simvp_spatio_kernel_enc=int(ckpt_args.get("simvp_spatio_kernel_enc", 3)),
        simvp_spatio_kernel_dec=int(ckpt_args.get("simvp_spatio_kernel_dec", 3)),
        simvp_mlp_ratio=float(ckpt_args.get("simvp_mlp_ratio", 8.0)),
        simvp_drop=float(ckpt_args.get("simvp_drop", 0.0)),
        simvp_drop_path=float(ckpt_args.get("simvp_drop_path", 0.0)),
        tau_spatio_kernel_enc=int(ckpt_args.get("tau_spatio_kernel_enc", 3)),
        tau_spatio_kernel_dec=int(ckpt_args.get("tau_spatio_kernel_dec", 3)),
        tau_mlp_ratio=float(ckpt_args.get("tau_mlp_ratio", 8.0)),
        tau_drop=float(ckpt_args.get("tau_drop", 0.0)),
        tau_drop_path=float(ckpt_args.get("tau_drop_path", 0.0)),
        convlstm_hidden=str(ckpt_args.get("convlstm_hidden", "128,128,128,128")),
        convlstm_filter_size=int(ckpt_args.get("convlstm_filter_size", 5)),
        convlstm_patch_size=int(ckpt_args.get("convlstm_patch_size", 4)),
        convlstm_stride=int(ckpt_args.get("convlstm_stride", 1)),
        convlstm_layer_norm=bool(ckpt_args.get("convlstm_layer_norm", False)),
        predrnnpp_hidden=str(ckpt_args.get("predrnnpp_hidden", "128,128,128,128")),
        predrnnpp_filter_size=int(ckpt_args.get("predrnnpp_filter_size", 5)),
        predrnnpp_patch_size=int(ckpt_args.get("predrnnpp_patch_size", 4)),
        predrnnpp_stride=int(ckpt_args.get("predrnnpp_stride", 1)),
        predrnnpp_layer_norm=bool(ckpt_args.get("predrnnpp_layer_norm", False)),
        predrnnpp_recipe=predrnnpp_recipe,
        predrnnpp_reverse_scheduled_sampling=bool(ckpt_args.get("reverse_scheduled_sampling", False)),
        arch=arch,
        hybrid_depth=int(ckpt_args.get("hybrid_depth", 2)),
        hybrid_heads=int(ckpt_args.get("hybrid_heads", 8)),
        hybrid_ffn_ratio=float(ckpt_args.get("hybrid_ffn_ratio", 4.0)),
        hybrid_attn_dropout=float(ckpt_args.get("hybrid_attn_dropout", 0.1)),
        hybrid_ffn_dropout=float(ckpt_args.get("hybrid_ffn_dropout", 0.1)),
        hybrid_drop_path=float(ckpt_args.get("hybrid_drop_path", 0.1)),
        use_local_branch=use_local_branch,
        local_crop=local_crop,
    )
    return model, {
        "in_T": in_T,
        "out_T": out_T,
        "arch": arch,
        "simvp_model_type": simvp_model_type,
        "predrnnpp_recipe": predrnnpp_recipe,
        "use_local_branch": use_local_branch,
        "local_crop": local_crop,
    }


@torch.no_grad()
def run_per_sample_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    skip_ssim: bool,
    strict_local: bool,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    total_seen = 0
    start = time.time()

    pbar = tqdm(loader, desc="Eval", dynamic_ncols=True)
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        x_local = batch.get("x_local")
        if x_local is not None:
            x_local = x_local.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(amp_enabled and device.type == "cuda")):
            pred = model(x, x_local=x_local, strict_local=strict_local)

        mae_pf = mae_per_frame(pred, y).detach().cpu()
        mse_pf = mse_per_frame(pred, y).detach().cpu()
        psnr_pf = psnr_from_mse(mse_pf).detach().cpu()
        if skip_ssim:
            ssim_pf = torch.full_like(mae_pf, float("nan"))
        else:
            ssim_pf = ssim_per_frame(pred, y).cpu()

        bs = x.shape[0]
        for i in range(bs):
            row = {
                "sample_id": str(batch["sample_ids"][i]),
                "dataset_idx": int(batch["dataset_indices"][i]),
                "mae_pred1": float(mae_pf[i, 0].item()),
                "mae_pred2": float(mae_pf[i, 1].item()),
                "mae_avg": float(mae_pf[i].mean().item()),
                "ssim_pred1": float(ssim_pf[i, 0].item()),
                "ssim_pred2": float(ssim_pf[i, 1].item()),
                "ssim_avg": float(ssim_pf[i].mean().item()),
                "psnr_pred1": float(psnr_pf[i, 0].item()),
                "psnr_pred2": float(psnr_pf[i, 1].item()),
                "psnr_avg": float(psnr_pf[i].mean().item()),
            }
            rows.append(row)

        total_seen += bs
        elapsed = time.time() - start
        sample_s = total_seen / max(elapsed, 1e-6)
        pbar.set_postfix(samples=total_seen, sample_s=f"{sample_s:.2f}")

    return rows


@torch.no_grad()
def export_selected_samples(
    model: nn.Module,
    dataset: IonogramManifestDataset,
    selected_rows: List[Dict[str, Any]],
    device: torch.device,
    amp_enabled: bool,
    strict_local: bool,
    mosaic_dir: Optional[Path],
    pred_frames_dir: Optional[Path],
    save_mosaic: bool,
    save_pred_only: bool,
) -> None:
    if not selected_rows:
        print("[Warn] No selected rows to export.")
        return

    model.eval()
    idxs = [int(r["dataset_idx"]) for r in selected_rows]

    for idx in tqdm(idxs, desc="Export selected", dynamic_ncols=True):
        item = dataset[idx]
        sample_id = str(item.get("sample_id", item.get("sequence_id", idx)))
        x = item["x"].unsqueeze(0).to(device)
        x_local = item.get("x_local")
        if x_local is not None:
            x_local = x_local.unsqueeze(0).to(device)
        y = item["y"]

        with torch.amp.autocast(device_type="cuda", enabled=(amp_enabled and device.type == "cuda")):
            pred = model(x, x_local=x_local, strict_local=strict_local).squeeze(0).detach().cpu()

        safe_sid = sanitize_name(sample_id)
        stem = f"idx_{idx:06d}__{safe_sid}"

        if save_mosaic and mosaic_dir is not None:
            mosaic = build_mosaic(item["x"], y, pred, sample_id=sample_id, dataset_idx=idx)
            mosaic.save(mosaic_dir / f"{stem}.png")

        if save_pred_only and pred_frames_dir is not None:
            sample_dir = pred_frames_dir / stem
            save_pred_frames(sample_dir, pred)


@torch.no_grad()
def export_all_pred_frames(
    model: nn.Module,
    dataset: IonogramManifestDataset,
    device: torch.device,
    amp_enabled: bool,
    strict_local: bool,
    save_dir: Path,
) -> None:
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(len(dataset)), desc="Export all pred frames", dynamic_ncols=True)
    for idx in pbar:
        item = dataset[idx]
        sample_id = str(item.get("sample_id", item.get("sequence_id", idx)))
        x = item["x"].unsqueeze(0).to(device)
        x_local = item.get("x_local")
        if x_local is not None:
            x_local = x_local.unsqueeze(0).to(device)

        with torch.amp.autocast(device_type="cuda", enabled=(amp_enabled and device.type == "cuda")):
            pred = model(x, x_local=x_local, strict_local=strict_local).squeeze(0).detach().cpu()

        safe_sid = sanitize_name(sample_id)
        stem = f"idx_{idx:06d}__{safe_sid}"
        sample_dir = save_dir / stem
        save_pred_frames(sample_dir, pred)


def main() -> None:
    args = parse_args()
    if args.no_pin_memory:
        args.pin_memory = False

    t0 = time.time()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    ckpt_args: Dict[str, Any] = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    image_mode = str(ckpt_args.get("image_mode", "L"))
    image_size = int(ckpt_args.get("image_size", 448))
    model, model_cfg = build_model_from_ckpt_args(ckpt_args, image_mode=image_mode, image_size=image_size)
    in_T = int(model_cfg["in_T"])
    out_T = int(model_cfg["out_T"])
    arch = str(model_cfg["arch"])
    predrnnpp_recipe = str(model_cfg.get("predrnnpp_recipe", "simvp"))
    use_local_branch = bool(model_cfg["use_local_branch"])
    local_crop = tuple(model_cfg["local_crop"])
    batch_size = int(coalesce(args.batch_size, ckpt_args.get("val_batch_size"), ckpt_args.get("batch_size"), default=4))
    num_workers = int(coalesce(args.num_workers, ckpt_args.get("num_workers"), default=4))

    if out_T != 2:
        raise ValueError(
            f"predict_all_preds.py currently expects out_T=2 for per-frame reporting/export, but the checkpoint uses out_T={out_T}."
        )

    device = choose_device(args.device)
    print(f"[Info] device={device}")
    print(f"[Info] image_mode={image_mode}, image_size={image_size}")
    print(f"[Info] arch={arch}, in_T={in_T}, out_T={out_T}, use_local_branch={use_local_branch}")
    if arch == "predrnnpp":
        print(f"[Info] predrnnpp_recipe={predrnnpp_recipe}")
    if use_local_branch:
        print(f"[Info] local_crop={local_crop}")

    output_dir = ensure_dir(args.output_dir)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / "per_sample_metrics.jsonl"
    selection_json_path = Path(args.selection_json_path) if args.selection_json_path else output_dir / "auto_selected_samples.json"

    ds_t0 = time.time()
    dataset = IonogramManifestDataset(
        manifest_path=args.val_manifest,
        image_mode=image_mode,
        image_size=image_size,
        local_crop=local_crop if use_local_branch else None,
    )
    attach_sample_metadata(dataset)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: int(args.max_samples)]
    validate_dataset_sequence_lengths(dataset, in_T=in_T, out_T=out_T)
    print(f"[Info] val samples={len(dataset)} (dataset ready in {time.time() - ds_t0:.1f}s)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_with_meta,
        drop_last=False,
    )
    print(f"[Info] dataloader batches={len(loader)}, batch_size={batch_size}, num_workers={num_workers}, amp={args.amp}")

    state_dict = extract_state_dict(ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch after auto-restoring configuration: "
            f"arch={arch}, in_T={in_T}, out_T={out_T}, use_local_branch={use_local_branch}, "
            f"local_crop={local_crop}."
        ) from exc
    model.to(device)
    print("[Info] model loaded from checkpoint")

    eval_t0 = time.time()
    rows = run_per_sample_eval(
        model=model,
        loader=loader,
        device=device,
        amp_enabled=args.amp,
        skip_ssim=args.skip_ssim,
        strict_local=use_local_branch,
    )
    jsonl_dump(rows, metrics_path)
    print(f"[Info] per-sample metrics saved: {metrics_path}")
    print(f"[Info] evaluation finished in {time.time() - eval_t0:.1f}s")

    sel_t0 = time.time()
    auto_selected = auto_select(
        rows,
        worst_k=args.worst_k,
        best_k=args.best_k,
        typical_k=args.typical_k,
        pred2_worse_k=args.pred2_worse_k,
    )
    json_dump(auto_selected, selection_json_path)
    print(f"[Info] auto selection saved: {selection_json_path}")
    print(f"[Info] selection finished in {time.time() - sel_t0:.1f}s")

    should_export_selected = args.save_selected or args.save_pred_frames
    if should_export_selected:
        if args.selected_json:
            selected_name, selected_rows = load_custom_selected(args.selected_json, rows, args.selected_name)
        else:
            selected_name, selected_rows = load_gallery_default(auto_selected)

        selected_rows_path = Path(args.selected_rows_path) if args.selected_rows_path else output_dir / f"selected_rows__{selected_name}.json"
        json_dump({"selected_name": selected_name, "rows": selected_rows}, selected_rows_path)
        print(f"[Info] selected rows saved: {selected_rows_path}")

        mosaic_dir = None
        if args.save_selected:
            mosaic_dir = ensure_dir(args.mosaic_dir) if args.mosaic_dir else ensure_dir(output_dir / f"mosaics__{selected_name}")
            print(f"[Info] mosaic dir: {mosaic_dir}")

        pred_frames_dir = None
        if args.save_pred_frames:
            pred_frames_dir = ensure_dir(args.pred_frames_dir) if args.pred_frames_dir else ensure_dir(output_dir / f"pred_frames__{selected_name}")
            print(f"[Info] pred frames dir: {pred_frames_dir}")

        export_t0 = time.time()
        export_selected_samples(
            model=model,
            dataset=dataset,
            selected_rows=selected_rows,
            device=device,
            amp_enabled=args.amp,
            strict_local=use_local_branch,
            mosaic_dir=mosaic_dir,
            pred_frames_dir=pred_frames_dir,
            save_mosaic=args.save_selected,
            save_pred_only=args.save_pred_frames,
        )
        print(f"[Info] selected export finished in {time.time() - export_t0:.1f}s")

    if args.save_all_pred_frames:
        all_pred_frames_dir = ensure_dir(args.all_pred_frames_dir) if args.all_pred_frames_dir else ensure_dir(output_dir / "pred_frames__all")
        print(f"[Info] all pred frames dir: {all_pred_frames_dir}")
        export_all_t0 = time.time()
        export_all_pred_frames(
            model=model,
            dataset=dataset,
            device=device,
            amp_enabled=args.amp,
            strict_local=use_local_branch,
            save_dir=all_pred_frames_dir,
        )
        print(f"[Info] all pred frames export finished in {time.time() - export_all_t0:.1f}s")

    print(f"[Info] done, total elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
