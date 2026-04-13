import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ionogram_manifest import IonogramManifestDataset
from simvp.wrapper import SUPPORTED_ARCHS, SimVPForecast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_mode", type=str, default=None, choices=["L", "RGB"])
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--in_T", type=int, default=None)
    parser.add_argument("--out_T", type=int, default=None)
    parser.add_argument("--arch", type=str, default=None, choices=SUPPORTED_ARCHS)
    parser.add_argument("--use_local_branch", action="store_true", default=None)
    parser.add_argument("--local_top", type=int, default=None)
    parser.add_argument("--local_bottom", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def collate_fn(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    x_local = None
    y_local = None
    if "x_local" in batch[0]:
        x_local = torch.stack([item["x_local"] for item in batch], dim=0)
    if "y_local" in batch[0]:
        y_local = torch.stack([item["y_local"] for item in batch], dim=0)
    return x, y, x_local, y_local


def crop_local_region(seq, top: int, bottom: int):
    return seq[:, :, :, top:bottom, :]


def tensor_mse(pred, target):
    return torch.mean((pred - target) ** 2)


def tensor_psnr(pred, target, data_range=1.0, eps=1e-8):
    mse_per_sample = ((pred - target) ** 2).flatten(1).mean(dim=1)
    psnr_per_sample = 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse_per_sample, min=eps))
    return psnr_per_sample.mean()


def batch_ssim_sum(pred, target, data_range=1.0):
    from skimage.metrics import structural_similarity as ssim

    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()

    batch_size, num_frames, channels, _, _ = pred_np.shape
    batch_sum = 0.0
    for batch_idx in range(batch_size):
        seq_scores = []
        for frame_idx in range(num_frames):
            if channels == 1:
                pred_frame = pred_np[batch_idx, frame_idx, 0]
                target_frame = target_np[batch_idx, frame_idx, 0]
                score = ssim(pred_frame, target_frame, data_range=data_range)
            else:
                pred_frame = np.transpose(pred_np[batch_idx, frame_idx], (1, 2, 0))
                target_frame = np.transpose(target_np[batch_idx, frame_idx], (1, 2, 0))
                score = ssim(pred_frame, target_frame, data_range=data_range, channel_axis=2)
            seq_scores.append(score)

        batch_sum += float(np.mean(seq_scores))
    return batch_sum


def resolve_override(cli_value, saved_args: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return saved_args.get(key, default)


def resolve_saved_first(saved_args: dict, key: str, cli_value, default):
    if key in saved_args:
        return saved_args[key]
    if cli_value is not None:
        return cli_value
    return default


def validate_dataset_sequence_lengths(dataset, in_T: int, out_T: int):
    if len(dataset) == 0:
        return

    sample = dataset[0]
    sample_in_T = int(sample["x"].shape[0])
    sample_out_T = int(sample["y"].shape[0])

    if sample_in_T != in_T or sample_out_T != out_T:
        raise ValueError(
            f"Dataset provides input/target lengths ({sample_in_T}, {sample_out_T}), but "
            f"the current configuration expects ({in_T}, {out_T})."
        )


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})

    image_mode = resolve_override(args.image_mode, saved_args, "image_mode", "L")
    image_size = resolve_override(args.image_size, saved_args, "image_size", 448)
    in_T = resolve_saved_first(saved_args, "in_T", args.in_T, 8)
    out_T = resolve_saved_first(saved_args, "out_T", args.out_T, 2)
    arch = resolve_override(args.arch, saved_args, "arch", "simvp")
    use_local_branch = resolve_override(args.use_local_branch, saved_args, "use_local_branch", False)
    local_top = resolve_saved_first(saved_args, "local_top", args.local_top, 186)
    local_bottom = resolve_saved_first(saved_args, "local_bottom", args.local_bottom, 410)
    channels = 1 if image_mode == "L" else 3

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and device.type == "cuda"

    model = SimVPForecast(
        in_T=in_T,
        out_T=out_T,
        C=channels,
        H=image_size,
        W=image_size,
        hid_S=saved_args.get("hid_S", 32),
        hid_T=saved_args.get("hid_T", 128),
        N_S=saved_args.get("N_S", 4),
        N_T=saved_args.get("N_T", 4),
        arch=arch,
        hybrid_depth=saved_args.get("hybrid_depth", 2),
        hybrid_heads=saved_args.get("hybrid_heads", 8),
        hybrid_ffn_ratio=saved_args.get("hybrid_ffn_ratio", 4.0),
        hybrid_attn_dropout=saved_args.get("hybrid_attn_dropout", 0.1),
        hybrid_ffn_dropout=saved_args.get("hybrid_ffn_dropout", 0.1),
        hybrid_drop_path=saved_args.get("hybrid_drop_path", 0.1),
        use_local_branch=use_local_branch,
        local_crop=(local_top, local_bottom),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    local_crop = (local_top, local_bottom)
    dataset = IonogramManifestDataset(
        manifest_path=args.manifest,
        image_mode=image_mode,
        image_size=image_size,
        local_crop=local_crop,
    )
    validate_dataset_sequence_lengths(dataset, in_T, out_T)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_local_mae = 0.0
    total_local_mse = 0.0
    total_count = 0
    total_local_count = 0

    with torch.inference_mode():
        for batch_idx, (x, y, x_local, y_local) in enumerate(tqdm(loader, desc="infer"), start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if x_local is not None:
                x_local = x_local.to(device, non_blocking=True)
            if y_local is not None:
                y_local = y_local.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                pred = model(x, x_local=x_local)

            mae = F.l1_loss(pred, y)
            mse = tensor_mse(pred, y)
            psnr = tensor_psnr(pred, y)
            ssim_sum = batch_ssim_sum(pred, y)
            if y_local is not None:
                pred_local = crop_local_region(pred, local_top, local_bottom)
                local_mae = F.l1_loss(pred_local, y_local)
                local_mse = tensor_mse(pred_local, y_local)
            else:
                local_mae = pred.new_zeros(())
                local_mse = pred.new_zeros(())

            if batch_idx == 1:
                print(f"arch: {arch}")
                print(f"input shape:  {tuple(x.shape)}")
                print(f"target shape: {tuple(y.shape)}")
                if x_local is not None:
                    print(f"x_local shape: {tuple(x_local.shape)}")
                if y_local is not None:
                    print(f"y_local shape: {tuple(y_local.shape)}")
                print(f"pred shape:   {tuple(pred.shape)}")

            batch_size = x.size(0)
            total_mae += mae.item() * batch_size
            total_mse += mse.item() * batch_size
            total_psnr += psnr.item() * batch_size
            total_ssim += ssim_sum
            total_count += batch_size
            if y_local is not None:
                total_local_mae += local_mae.item() * batch_size
                total_local_mse += local_mse.item() * batch_size
                total_local_count += batch_size

    if total_count == 0:
        raise RuntimeError("No samples were processed. Check --max_batches or the manifest file.")

    print(f"checkpoint: {ckpt_path}")
    print(f"samples: {total_count}")
    print(f"global_mae:  {total_mae / total_count:.6f}")
    print(f"global_mse:  {total_mse / total_count:.6f}")
    print(f"global_psnr: {total_psnr / total_count:.6f}")
    print(f"global_ssim: {total_ssim / total_count:.6f}")
    print(f"local_mae:   {total_local_mae / max(total_local_count, 1):.6f}")
    print(f"local_mse:   {total_local_mse / max(total_local_count, 1):.6f}")


if __name__ == "__main__":
    main()
