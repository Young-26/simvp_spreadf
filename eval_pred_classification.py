#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate exported prediction frames with a classification model.

This script is designed for the two-layer export structure produced by the
modified predict_all_preds.py:

output_dir/
  pred_frames_manifest.jsonl
  cls_eval/
    pred1/
      Freq/
      Mix/
      Range/
      SRange/
      non-SF/
    pred2/
      Freq/
      Mix/
      Range/
      SRange/
      non-SF/

It loads the classifier using the same preprocessing style as the repo's
predict.py (Resize to 448, ImageNet normalization, ImageFolder-like class
folders), then computes:
- overall multiclass accuracy for pred1 / pred2 / avg
- by-year multiclass accuracy for pred1 / pred2 / avg
- overall and by-year occurrence accuracy
- overall and by-year F1-score (macro / weighted)
- detailed per-image prediction CSVs
- confusion matrices (multiclass + occurrence)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ResNetmodel import resnet50


EXPECTED_CLASSES = ["Freq", "Mix", "Range", "SRange", "non-SF"]
SPREAD_F_CLASSES = {"Freq", "Mix", "Range", "SRange"}
NON_SPREAD_F_CLASS = "non-SF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported Pred1/Pred2 classification results")
    parser.add_argument("--pred1_dir", type=str, required=True, help="cls_eval/pred1 directory")
    parser.add_argument("--pred2_dir", type=str, required=True, help="cls_eval/pred2 directory")
    parser.add_argument("--class_indices_json", type=str, required=True, help="Path to class_indices.json")
    parser.add_argument("--weights_path", type=str, required=True, help="Classifier weights path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV/JSON results")
    parser.add_argument(
        "--pred_frames_manifest",
        type=str,
        default=None,
        help="Optional pred_frames_manifest.jsonl. If omitted, try to auto-find it next to cls_eval/.",
    )
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_logits", action="store_true", help="Save max probability/logit columns to CSV")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_class_names(json_path: str | Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names = [mapping[str(i)] for i in range(len(mapping))]
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"class_indices.json classes {class_names} do not match expected {EXPECTED_CLASSES}."
        )
    return class_names


class OrderedClassFolderDataset(Dataset):
    def __init__(self, root: str | Path, class_names: Sequence[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.transform = transform
        self.samples: List[Tuple[Path, int, str]] = []
        self._scan()

    def _scan(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            files = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                ]
            )
            for p in files:
                self.samples.append((p, class_idx, class_name))
        if not self.samples:
            raise RuntimeError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label_idx, label_name = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": label_idx,
            "label_name": label_name,
            "path": str(path),
            "filename": path.name,
        }


class ManifestLookup:
    def __init__(self, manifest_path: Optional[str | Path]):
        self.by_dataset_split: Dict[Tuple[int, str], Dict] = {}
        self.by_sample_split: Dict[Tuple[str, str], Dict] = {}
        self.manifest_path = Path(manifest_path) if manifest_path else None
        if self.manifest_path is not None and self.manifest_path.exists():
            self._load(self.manifest_path)

    def _load(self, manifest_path: Path) -> None:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                dataset_idx = int(row.get("dataset_idx", -1))
                sample_id = str(row.get("sample_id", ""))
                self.by_dataset_split[(dataset_idx, "pred1")] = row
                self.by_dataset_split[(dataset_idx, "pred2")] = row
                if sample_id:
                    self.by_sample_split[(sample_id, "pred1")] = row
                    self.by_sample_split[(sample_id, "pred2")] = row

    def lookup(self, filename: str, split_name: str) -> Dict:
        dataset_idx, sample_id = parse_export_filename(filename)
        if dataset_idx is not None and (dataset_idx, split_name) in self.by_dataset_split:
            return self.by_dataset_split[(dataset_idx, split_name)]
        if sample_id is not None and (sample_id, split_name) in self.by_sample_split:
            return self.by_sample_split[(sample_id, split_name)]
        return {}


def auto_find_manifest(pred1_dir: str | Path, explicit_manifest: Optional[str | Path]) -> Optional[Path]:
    if explicit_manifest:
        p = Path(explicit_manifest)
        return p if p.exists() else None
    pred1_dir = Path(pred1_dir).resolve()
    candidates = [
        pred1_dir.parent.parent / "pred_frames_manifest.jsonl",
        pred1_dir.parent.parent.parent / "pred_frames_manifest.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_export_filename(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse names like idx_000123__20130926131500__pred1.png."""
    m = re.match(r"^idx_(\d+)__(.+?)__pred[12]\.[^.]+$", filename)
    if not m:
        return None, None
    dataset_idx = int(m.group(1))
    sample_id = m.group(2)
    return dataset_idx, sample_id


def infer_year_from_filename(filename: str) -> Optional[int]:
    _dataset_idx, sample_id = parse_export_filename(filename)
    if sample_id is None:
        return None
    m = re.match(r"^(19|20)\d{2}", sample_id)
    if m:
        return int(sample_id[:4])
    m2 = re.search(r"(19|20)\d{2}", sample_id)
    if m2:
        return int(m2.group(0))
    return None


def choose_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_model(weights_path: str | Path, num_classes: int, device: torch.device):
    model = resnet50(num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(state)}")

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded weights: {weights_path}")
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
    model.eval()
    return model


def collate_fn(batch: List[Dict]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    return {
        "images": images,
        "labels": labels,
        "label_names": [b["label_name"] for b in batch],
        "paths": [b["path"] for b in batch],
        "filenames": [b["filename"] for b in batch],
    }


def occurrence_label(label_name: str) -> str:
    return "Spread-F" if label_name in SPREAD_F_CLASSES else "non-Spread-F"


def evaluate_split(
    split_name: str,
    split_dir: str | Path,
    model: torch.nn.Module,
    class_names: Sequence[str],
    device: torch.device,
    img_size: int,
    batch_size: int,
    num_workers: int,
    manifest_lookup: ManifestLookup,
    save_logits: bool,
) -> List[Dict]:
    dataset = OrderedClassFolderDataset(split_dir, class_names=class_names, transform=build_transform(img_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    rows: List[Dict] = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            logits = model(images)
            probs = softmax(logits)
            pred_idx = torch.argmax(logits, dim=1)

            logits_np = logits.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            pred_idx_np = pred_idx.detach().cpu().numpy()
            gt_idx_np = batch["labels"].cpu().numpy()

            for i in range(len(batch["filenames"])):
                filename = batch["filenames"][i]
                path = batch["paths"][i]
                gt_idx = int(gt_idx_np[i])
                pd_idx = int(pred_idx_np[i])
                gt_label = class_names[gt_idx]
                pred_label = class_names[pd_idx]

                manifest_row = manifest_lookup.lookup(filename, split_name)
                dataset_idx, sample_id = parse_export_filename(filename)
                year = manifest_row.get("year") if manifest_row else None
                if year is None:
                    year = infer_year_from_filename(filename)

                row = {
                    "split": split_name,
                    "path": path,
                    "filename": filename,
                    "dataset_idx": dataset_idx,
                    "sample_id": manifest_row.get("sample_id") if manifest_row else sample_id,
                    "sequence_id": manifest_row.get("sequence_id") if manifest_row else None,
                    "year": int(year) if year is not None else None,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "gt_idx": gt_idx,
                    "pred_idx": pd_idx,
                    "correct": int(gt_idx == pd_idx),
                    "gt_occurrence": occurrence_label(gt_label),
                    "pred_occurrence": occurrence_label(pred_label),
                    "occurrence_correct": int(occurrence_label(gt_label) == occurrence_label(pred_label)),
                }
                if save_logits:
                    row["max_prob"] = float(np.max(probs_np[i]))
                    row["max_logit"] = float(np.max(logits_np[i]))
                rows.append(row)

    return rows


def accuracy_from_rows(rows: Sequence[Dict], key: str = "correct") -> float:
    if not rows:
        return 0.0
    return float(sum(int(r[key]) for r in rows) / len(rows))


def f1_from_rows(rows: Sequence[Dict], average: str, occurrence: bool = False) -> float:
    if not rows:
        return 0.0
    if occurrence:
        labels = ["Spread-F", "non-Spread-F"]
        y_true = [r["gt_occurrence"] for r in rows]
        y_pred = [r["pred_occurrence"] for r in rows]
    else:
        labels = EXPECTED_CLASSES
        y_true = [r["gt_label"] for r in rows]
        y_pred = [r["pred_label"] for r in rows]
    return float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0))


def confusion_rows(rows: Sequence[Dict], occurrence: bool = False) -> Tuple[List[str], np.ndarray]:
    if occurrence:
        labels = ["Spread-F", "non-Spread-F"]
        y_true = [r["gt_occurrence"] for r in rows]
        y_pred = [r["pred_occurrence"] for r in rows]
    else:
        labels = EXPECTED_CLASSES
        y_true = [r["gt_label"] for r in rows]
        y_pred = [r["pred_label"] for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return labels, cm


def write_confusion_csv(path: str | Path, labels: Sequence[str], cm: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *labels])
        for label, row in zip(labels, cm.tolist()):
            writer.writerow([label, *row])


def write_rows_csv(path: str | Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "split",
        "path",
        "filename",
        "dataset_idx",
        "sample_id",
        "sequence_id",
        "year",
        "gt_label",
        "pred_label",
        "gt_idx",
        "pred_idx",
        "correct",
        "gt_occurrence",
        "pred_occurrence",
        "occurrence_correct",
        "max_prob",
        "max_logit",
    ]
    active_fields = [fn for fn in fieldnames if any(fn in r for r in rows)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=active_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in active_fields})


def compute_split_metrics(rows: Sequence[Dict]) -> Dict:
    return {
        "n": len(rows),
        "accuracy": accuracy_from_rows(rows, "correct"),
        "occurrence_accuracy": accuracy_from_rows(rows, "occurrence_correct"),
        "macro_f1": f1_from_rows(rows, average="macro", occurrence=False),
        "weighted_f1": f1_from_rows(rows, average="weighted", occurrence=False),
        "occurrence_macro_f1": f1_from_rows(rows, average="macro", occurrence=True),
        "occurrence_weighted_f1": f1_from_rows(rows, average="weighted", occurrence=True),
    }


def compute_overall_summary(pred1_rows: Sequence[Dict], pred2_rows: Sequence[Dict]) -> Dict:
    pred1 = compute_split_metrics(pred1_rows)
    pred2 = compute_split_metrics(pred2_rows)
    return {
        "pred1": pred1,
        "pred2": pred2,
        "avg": {
            "accuracy": (pred1["accuracy"] + pred2["accuracy"]) / 2.0,
            "occurrence_accuracy": (pred1["occurrence_accuracy"] + pred2["occurrence_accuracy"]) / 2.0,
            "macro_f1": (pred1["macro_f1"] + pred2["macro_f1"]) / 2.0,
            "weighted_f1": (pred1["weighted_f1"] + pred2["weighted_f1"]) / 2.0,
            "occurrence_macro_f1": (pred1["occurrence_macro_f1"] + pred2["occurrence_macro_f1"]) / 2.0,
            "occurrence_weighted_f1": (pred1["occurrence_weighted_f1"] + pred2["occurrence_weighted_f1"]) / 2.0,
        },
        "combined_all_frames": {
            "n": len(pred1_rows) + len(pred2_rows),
            "accuracy": accuracy_from_rows(list(pred1_rows) + list(pred2_rows), "correct"),
            "occurrence_accuracy": accuracy_from_rows(list(pred1_rows) + list(pred2_rows), "occurrence_correct"),
            "macro_f1": f1_from_rows(list(pred1_rows) + list(pred2_rows), average="macro", occurrence=False),
            "weighted_f1": f1_from_rows(list(pred1_rows) + list(pred2_rows), average="weighted", occurrence=False),
            "occurrence_macro_f1": f1_from_rows(list(pred1_rows) + list(pred2_rows), average="macro", occurrence=True),
            "occurrence_weighted_f1": f1_from_rows(list(pred1_rows) + list(pred2_rows), average="weighted", occurrence=True),
        },
    }


def compute_by_year(pred1_rows: Sequence[Dict], pred2_rows: Sequence[Dict]) -> List[Dict]:
    by_year_1: Dict[int, List[Dict]] = defaultdict(list)
    by_year_2: Dict[int, List[Dict]] = defaultdict(list)
    for row in pred1_rows:
        if row.get("year") is not None:
            by_year_1[int(row["year"])].append(row)
    for row in pred2_rows:
        if row.get("year") is not None:
            by_year_2[int(row["year"])].append(row)

    years = sorted(set(by_year_1.keys()) | set(by_year_2.keys()))
    rows: List[Dict] = []
    for year in years:
        r1 = by_year_1.get(year, [])
        r2 = by_year_2.get(year, [])
        m1 = compute_split_metrics(r1) if r1 else None
        m2 = compute_split_metrics(r2) if r2 else None

        pred1_acc = m1["accuracy"] if m1 else None
        pred2_acc = m2["accuracy"] if m2 else None
        pred1_occ = m1["occurrence_accuracy"] if m1 else None
        pred2_occ = m2["occurrence_accuracy"] if m2 else None
        pred1_macro_f1 = m1["macro_f1"] if m1 else None
        pred2_macro_f1 = m2["macro_f1"] if m2 else None
        pred1_weighted_f1 = m1["weighted_f1"] if m1 else None
        pred2_weighted_f1 = m2["weighted_f1"] if m2 else None

        row = {
            "year": year,
            "n_pred1": len(r1),
            "n_pred2": len(r2),
            "pred1_accuracy": pred1_acc,
            "pred2_accuracy": pred2_acc,
            "avg_accuracy": None if pred1_acc is None or pred2_acc is None else (pred1_acc + pred2_acc) / 2.0,
            "pred1_occurrence_accuracy": pred1_occ,
            "pred2_occurrence_accuracy": pred2_occ,
            "avg_occurrence_accuracy": None if pred1_occ is None or pred2_occ is None else (pred1_occ + pred2_occ) / 2.0,
            "pred1_macro_f1": pred1_macro_f1,
            "pred2_macro_f1": pred2_macro_f1,
            "avg_macro_f1": None if pred1_macro_f1 is None or pred2_macro_f1 is None else (pred1_macro_f1 + pred2_macro_f1) / 2.0,
            "pred1_weighted_f1": pred1_weighted_f1,
            "pred2_weighted_f1": pred2_weighted_f1,
            "avg_weighted_f1": None if pred1_weighted_f1 is None or pred2_weighted_f1 is None else (pred1_weighted_f1 + pred2_weighted_f1) / 2.0,
        }
        rows.append(row)
    return rows


def write_year_csv(path: str | Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "year",
        "n_pred1",
        "n_pred2",
        "pred1_accuracy",
        "pred2_accuracy",
        "avg_accuracy",
        "pred1_occurrence_accuracy",
        "pred2_occurrence_accuracy",
        "avg_occurrence_accuracy",
        "pred1_macro_f1",
        "pred2_macro_f1",
        "avg_macro_f1",
        "pred1_weighted_f1",
        "pred2_weighted_f1",
        "avg_weighted_f1",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    class_names = load_class_names(args.class_indices_json)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    manifest_path = auto_find_manifest(args.pred1_dir, args.pred_frames_manifest)
    if manifest_path is not None:
        print(f"Using manifest lookup: {manifest_path}")
    else:
        print("[WARN] pred_frames_manifest.jsonl not found; falling back to parsing year from filename.")
    manifest_lookup = ManifestLookup(manifest_path)

    model = load_model(args.weights_path, num_classes=len(class_names), device=device)

    pred1_rows = evaluate_split(
        split_name="pred1",
        split_dir=args.pred1_dir,
        model=model,
        class_names=class_names,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_lookup=manifest_lookup,
        save_logits=args.save_logits,
    )
    pred2_rows = evaluate_split(
        split_name="pred2",
        split_dir=args.pred2_dir,
        model=model,
        class_names=class_names,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_lookup=manifest_lookup,
        save_logits=args.save_logits,
    )

    write_rows_csv(output_dir / "pred1_predictions.csv", pred1_rows)
    write_rows_csv(output_dir / "pred2_predictions.csv", pred2_rows)

    overall = compute_overall_summary(pred1_rows, pred2_rows)
    with open(output_dir / "metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    by_year = compute_by_year(pred1_rows, pred2_rows)
    write_year_csv(output_dir / "metrics_by_year.csv", by_year)

    labels_mc1, cm_mc1 = confusion_rows(pred1_rows, occurrence=False)
    labels_oc1, cm_oc1 = confusion_rows(pred1_rows, occurrence=True)
    labels_mc2, cm_mc2 = confusion_rows(pred2_rows, occurrence=False)
    labels_oc2, cm_oc2 = confusion_rows(pred2_rows, occurrence=True)

    write_confusion_csv(output_dir / "pred1_confusion_matrix.csv", labels_mc1, cm_mc1)
    write_confusion_csv(output_dir / "pred1_occurrence_confusion_matrix.csv", labels_oc1, cm_oc1)
    write_confusion_csv(output_dir / "pred2_confusion_matrix.csv", labels_mc2, cm_mc2)
    write_confusion_csv(output_dir / "pred2_occurrence_confusion_matrix.csv", labels_oc2, cm_oc2)

    print("\nDone.")
    print(f"Saved: {output_dir / 'pred1_predictions.csv'}")
    print(f"Saved: {output_dir / 'pred2_predictions.csv'}")
    print(f"Saved: {output_dir / 'metrics_overall.json'}")
    print(f"Saved: {output_dir / 'metrics_by_year.csv'}")
    print(f"Saved: {output_dir / 'pred1_confusion_matrix.csv'}")
    print(f"Saved: {output_dir / 'pred2_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
