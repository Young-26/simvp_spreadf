import os
import csv
import time
import math
import argparse
import logging
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.ionogram_manifest import IonogramManifestDataset
from simvp.wrapper import SUPPORTED_ARCHS, SimVPForecast
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)

    parser.add_argument("--image_mode", type=str, default="L", choices=["L", "RGB"])
    parser.add_argument("--image_size", type=int, default=448)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hid_S", type=int, default=32)
    parser.add_argument("--hid_T", type=int, default=128)
    parser.add_argument("--N_S", type=int, default=4)
    parser.add_argument("--N_T", type=int, default=4)
    parser.add_argument("--in_T", type=int, default=8)
    parser.add_argument("--out_T", type=int, default=2)
    parser.add_argument("--arch", type=str, default="simvp", choices=SUPPORTED_ARCHS)
    parser.add_argument("--hybrid_depth", type=int, default=2)
    parser.add_argument("--hybrid_heads", type=int, default=8)
    parser.add_argument("--hybrid_ffn_ratio", type=float, default=4.0)
    parser.add_argument("--hybrid_attn_dropout", type=float, default=0.1)
    parser.add_argument("--hybrid_ffn_dropout", type=float, default=0.1)
    parser.add_argument("--hybrid_drop_path", type=float, default=0.1)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./work_dirs/simvp_spreadf")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def validate_dataset_sequence_lengths(dataset, split_name: str, in_T: int, out_T: int):
    if len(dataset) == 0:
        return

    sample = dataset[0]
    sample_in_T = int(sample["x"].shape[0])
    sample_out_T = int(sample["y"].shape[0])

    if sample_in_T != in_T or sample_out_T != out_T:
        raise ValueError(
            f"{split_name} dataset provides input/target lengths ({sample_in_T}, {sample_out_T}), "
            f"but the current configuration expects ({in_T}, {out_T})."
        )


def collate_fn(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    return x, y


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """
    torchrun 启动时会自动注入:
    RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_sum_scalar(value, device):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("simvp_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(
        os.path.join(save_dir, "train.log"),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def init_csv(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_mae",
                "val_mae",
                "val_mse",
                "val_psnr",
                "val_ssim",
                "lr",
                "epoch_time",
                "gpu_mem_mb",
                "best_epoch",
                "best_val_mae",
            ])


def append_csv(csv_path: str, row: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["epoch"],
            f'{row["train_mae"]:.8f}',
            f'{row["val_mae"]:.8f}',
            f'{row["val_mse"]:.8f}',
            f'{row["val_psnr"]:.8f}',
            f'{row["val_ssim"]:.8f}',
            f'{row["lr"]:.10f}',
            f'{row["epoch_time"]:.4f}',
            f'{row["gpu_mem_mb"]:.2f}',
            row["best_epoch"],
            f'{row["best_val_mae"]:.8f}',
        ])


def tensor_mse(pred, target):
    return torch.mean((pred - target) ** 2)


def tensor_psnr(pred, target, data_range=1.0, eps=1e-8):
    mse_per_sample = ((pred - target) ** 2).flatten(1).mean(dim=1)
    psnr_per_sample = 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse_per_sample, min=eps))
    return psnr_per_sample.mean()


def batch_ssim_sum(pred, target, data_range=1.0):
    """
    返回当前 batch 的 SSIM 累加和，而不是平均值
    这样方便跨卡 all_reduce 后再除总样本数
    pred, target: [B, T, C, H, W]
    """
    from skimage.metrics import structural_similarity as ssim

    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()

    B, T, C, H, W = pred_np.shape
    batch_sum = 0.0

    for b in range(B):
        seq_scores = []
        for t in range(T):
            if C == 1:
                p = pred_np[b, t, 0]
                g = target_np[b, t, 0]
                score = ssim(p, g, data_range=data_range)
            else:
                p = np.transpose(pred_np[b, t], (1, 2, 0))
                g = np.transpose(target_np[b, t], (1, 2, 0))
                score = ssim(p, g, data_range=data_range, channel_axis=2)
            seq_scores.append(score)

        batch_sum += float(np.mean(seq_scores))

    return batch_sum


@torch.no_grad()
def evaluate(model, loader, mae_criterion, device, amp_enabled=False):
    model.eval()

    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_count = 0.0

    iterator = tqdm(loader, desc="val", leave=False) if is_main_process() else loader

    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(x)
            mae = mae_criterion(pred, y)

        mse = tensor_mse(pred, y).item()
        psnr = tensor_psnr(pred, y).item()
        ssim_sum = batch_ssim_sum(pred, y)
        bs = x.size(0)

        total_mae += mae.item() * bs
        total_mse += mse * bs
        total_psnr += psnr * bs
        total_ssim += ssim_sum
        total_count += bs

    total_mae = reduce_sum_scalar(total_mae, device)
    total_mse = reduce_sum_scalar(total_mse, device)
    total_psnr = reduce_sum_scalar(total_psnr, device)
    total_ssim = reduce_sum_scalar(total_ssim, device)
    total_count = reduce_sum_scalar(total_count, device)

    metrics = {
        "val_mae": total_mae / max(total_count, 1.0),
        "val_mse": total_mse / max(total_count, 1.0),
        "val_psnr": total_psnr / max(total_count, 1.0),
        "val_ssim": total_ssim / max(total_count, 1.0),
    }
    return metrics


def save_checkpoint(path, epoch, model, optimizer, scaler, args, best_epoch, best_val_mae, history, status):
    raw_model = unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "history": history,
        "status": status,
    }
    torch.save(ckpt, path)


def write_report(report_path, status, reason, total_epochs, completed_epochs, best_epoch, best_val_mae, history):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SimVP Training Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"status: {status}\n")
        f.write(f"reason: {reason}\n")
        f.write(f"total_epochs: {total_epochs}\n")
        f.write(f"completed_epochs: {completed_epochs}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_mae: {best_val_mae:.8f}\n")

        if len(history) > 0:
            last = history[-1]
            f.write("\nLast Epoch Metrics\n")
            f.write("-" * 60 + "\n")
            for k, v in last.items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.8f}\n")
                else:
                    f.write(f"{k}: {v}\n")


def main():
    args = parse_args()

    use_ddp, rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed + rank)

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_dir = os.path.join(args.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        logger = setup_logger(args.save_dir)
        csv_path = os.path.join(args.save_dir, "metrics.csv")
        report_path = os.path.join(args.save_dir, "train_report.txt")
        init_csv(csv_path)

        logger.info("Starting training")
        logger.info(f"save_dir: {args.save_dir}")
        logger.info(f"arch: {args.arch}")
        logger.info(f"in_T: {args.in_T}  out_T: {args.out_T}")
        logger.info(f"train_manifest: {args.train_manifest}")
        logger.info(f"val_manifest: {args.val_manifest}")
        logger.info(f"use_ddp: {use_ddp}")
        logger.info(f"world_size: {world_size}")
    else:
        logger = None
        csv_path = None
        report_path = None
        ckpt_dir = os.path.join(args.save_dir, "checkpoints")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if use_ddp else args.device)
    else:
        device = torch.device("cpu")
    amp_enabled = args.amp and device.type == "cuda"

    if is_main_process():
        logger.info(f"device: {device}")
        logger.info(f"amp_enabled: {amp_enabled}")
        if args.arch == "hybrid_unet_facts":
            logger.info("hybrid_unet_facts uses a strict Fac-T-S translator plus future forecasting heads.")

    train_set = IonogramManifestDataset(
        manifest_path=args.train_manifest,
        image_mode=args.image_mode,
        image_size=args.image_size,
    )
    val_set = IonogramManifestDataset(
        manifest_path=args.val_manifest,
        image_mode=args.image_mode,
        image_size=args.image_size,
    )

    if is_main_process():
        logger.info(f"train_samples: {len(train_set)}")
        logger.info(f"val_samples: {len(val_set)}")

    validate_dataset_sequence_lengths(train_set, "train", args.in_T, args.out_T)
    validate_dataset_sequence_lengths(val_set, "val", args.in_T, args.out_T)

    channels = 1 if args.image_mode == "L" else 3

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if use_ddp else None

    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if use_ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = SimVPForecast(
        in_T=args.in_T,
        out_T=args.out_T,
        C=channels,
        H=args.image_size,
        W=args.image_size,
        hid_S=args.hid_S,
        hid_T=args.hid_T,
        N_S=args.N_S,
        N_T=args.N_T,
        arch=args.arch,
        hybrid_depth=args.hybrid_depth,
        hybrid_heads=args.hybrid_heads,
        hybrid_ffn_ratio=args.hybrid_ffn_ratio,
        hybrid_attn_dropout=args.hybrid_attn_dropout,
        hybrid_ffn_dropout=args.hybrid_ffn_dropout,
        hybrid_drop_path=args.hybrid_drop_path,
    ).to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    mae_criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_epoch = 0
    best_val_mae = float("inf")
    history = []

    status = "RUNNING"
    reason = ""
    completed_epochs = 0

    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            model.train()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_mae_sum = 0.0
            train_count = 0.0

            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}") if is_main_process() else train_loader

            for x, y in iterator:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    pred = model(x)
                    loss = mae_criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = x.size(0)
                train_mae_sum += loss.item() * bs
                train_count += bs

                if is_main_process():
                    iterator.set_postfix(mae=f"{loss.item():.6f}")

            train_mae_sum = reduce_sum_scalar(train_mae_sum, device)
            train_count = reduce_sum_scalar(train_count, device)
            train_mae = train_mae_sum / max(train_count, 1.0)

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                mae_criterion=mae_criterion,
                device=device,
                amp_enabled=amp_enabled,
            )

            val_mae = val_metrics["val_mae"]
            val_mse = val_metrics["val_mse"]
            val_psnr = val_metrics["val_psnr"]
            val_ssim = val_metrics["val_ssim"]

            lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start_time

            if device.type == "cuda":
                gpu_mem_mb_local = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
            else:
                gpu_mem_mb_local = 0.0

            gpu_mem_mb = reduce_sum_scalar(gpu_mem_mb_local, device) / get_world_size()

            if is_main_process():
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    is_best = True
                else:
                    is_best = False

                row = {
                    "epoch": epoch,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "lr": lr,
                    "epoch_time": epoch_time,
                    "gpu_mem_mb": gpu_mem_mb,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                }
                history.append(row)
                append_csv(csv_path, row)

                save_checkpoint(
                    path=os.path.join(ckpt_dir, "last.ckpt"),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    best_epoch=best_epoch,
                    best_val_mae=best_val_mae,
                    history=history,
                    status="RUNNING",
                )

                logger.info(f"[Epoch {epoch}/{args.epochs}]")
                logger.info(
                    f"train_mae: {train_mae:.4f}  val_mae: {val_mae:.4f}"
                )
                logger.info(
                    f"val_mse: {val_mse:.4f}  val_psnr: {val_psnr:.4f}  val_ssim: {val_ssim:.4f}"
                )
                logger.info(
                    f"lr: {lr:.6f}  epoch_time: {epoch_time:.1f}s  gpu_mem: {gpu_mem_mb:.0f}MB"
                )
                logger.info(
                    f"best_epoch: {best_epoch}  best_val_mae: {best_val_mae:.4f}"
                )

                if is_best:
                    best_path = os.path.join(ckpt_dir, "best.ckpt")
                    save_checkpoint(
                        path=best_path,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        args=args,
                        best_epoch=best_epoch,
                        best_val_mae=best_val_mae,
                        history=history,
                        status="BEST",
                    )
                    logger.info(f"Found better model by val_mae. Saving to {best_path}")

            completed_epochs = epoch

        status = "FINISHED"
        reason = "Training completed normally."

    except KeyboardInterrupt:
        status = "INTERRUPTED"
        reason = "KeyboardInterrupt"
        if is_main_process():
            logger.warning("Training interrupted by user (KeyboardInterrupt).")

    except Exception as e:
        status = "FAILED"
        reason = f"{type(e).__name__}: {str(e)}"
        if is_main_process():
            logger.exception("Training failed with exception:")
            traceback.print_exc()

    finally:
        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            try:
                if "model" in locals():
                    last_path = os.path.join(ckpt_dir, "last.ckpt")
                    save_checkpoint(
                        path=last_path,
                        epoch=completed_epochs,
                        model=model,
                        optimizer=optimizer if "optimizer" in locals() else None,
                        scaler=scaler if "scaler" in locals() else None,
                        args=args,
                        best_epoch=best_epoch,
                        best_val_mae=best_val_mae,
                        history=history,
                        status=status,
                    )
                    logger.info(f"Last checkpoint saved to {last_path}")
            except Exception as save_e:
                logger.exception(f"Failed to save last checkpoint in finally: {save_e}")

            try:
                write_report(
                    report_path=report_path,
                    status=status,
                    reason=reason,
                    total_epochs=args.epochs,
                    completed_epochs=completed_epochs,
                    best_epoch=best_epoch,
                    best_val_mae=best_val_mae if math.isfinite(best_val_mae) else -1.0,
                    history=history,
                )
                logger.info(f"Training report written to {report_path}")
            except Exception as report_e:
                logger.exception(f"Failed to write training report: {report_e}")

            logger.info(f"Training finished with status: {status}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Completed epochs: {completed_epochs}/{args.epochs}")
            logger.info(f"Best epoch: {best_epoch}")
            if math.isfinite(best_val_mae):
                logger.info(f"Best val_mae: {best_val_mae:.6f}")

        cleanup_distributed()


if __name__ == "__main__":
    main()
