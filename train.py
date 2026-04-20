import os
import csv
import time
import math
import sys
import argparse
import logging
import traceback
from contextlib import nullcontext
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torchvision.models import VGG16_Weights, vgg16

from datasets.ionogram_manifest import IonogramManifestDataset
from simvp.simvp_config import (
    PREDRNNPP_RECIPE_CHOICES,
    SIMVP_MODEL_TYPE_ALIASES,
    SIMVP_OPENSTL_TRAIN_PRESET,
    SIMVP_RECIPE_CHOICES,
    get_effective_simvp_recipe,
    is_simvp_openstl_recipe,
    normalize_predrnnpp_recipe,
    normalize_simvp_model_type,
    normalize_simvp_recipe,
)
from simvp.tau_model import tau_diff_div_reg
from simvp.wrapper import SUPPORTED_ARCHS, SimVPForecast
from utils.seed import set_seed


def get_predrnnpp_recipe(args) -> str:
    return normalize_predrnnpp_recipe(getattr(args, "predrnnpp_recipe", "simvp"))


def get_simvp_model_type(args) -> str:
    return normalize_simvp_model_type(getattr(args, "simvp_model_type", "incepu"))


def get_simvp_recipe(args) -> str:
    return normalize_simvp_recipe(getattr(args, "simvp_recipe", "auto"))


def get_effective_simvp_recipe_from_args(args) -> str:
    return get_effective_simvp_recipe(
        getattr(args, "arch", "simvp"),
        get_simvp_model_type(args),
        get_simvp_recipe(args),
    )


def is_predrnnpp_arch(args) -> bool:
    return str(args.arch).lower() == "predrnnpp"


def is_tau_arch(args) -> bool:
    return str(args.arch).lower() == "tau"


def uses_predrnnpp_openstl_recipe(args) -> bool:
    return is_predrnnpp_arch(args) and get_predrnnpp_recipe(args) == "openstl"


def uses_simvp_openstl_recipe(args) -> bool:
    return is_simvp_openstl_recipe(
        getattr(args, "arch", "simvp"),
        getattr(args, "simvp_model_type", "incepu"),
        getattr(args, "simvp_recipe", "auto"),
    )


def uses_simvp_moganet_openstl_loss(args) -> bool:
    return (
        str(getattr(args, "arch", "simvp")).lower() == "simvp"
        and get_simvp_model_type(args) == "moganet"
        and get_effective_simvp_recipe_from_args(args) == "openstl"
    )


def uses_earthfarseer_openstl_loss(args) -> bool:
    # EarthFarseer only reuses the OpenSTL-style MSE objective; it does not
    # consume PredRNN++'s full-sequence/mask_true training interface.
    return str(getattr(args, "arch", "simvp")).lower() == "earthfarseer"


def uses_predformer_facts_openstl_loss(args) -> bool:
    # PredFormer_FacTS follows an OpenSTL-style direct prediction objective. The
    # concrete loss type is selected via --predformer_loss.
    return str(getattr(args, "arch", "simvp")).lower() == "predformer_facts"


def get_predformer_loss(args) -> str:
    loss_name = str(getattr(args, "predformer_loss", "mae")).strip().lower()
    if loss_name not in ("mae", "mse"):
        raise ValueError(
            f"Unsupported predformer_loss '{loss_name}'. Available choices: ('mae', 'mse')."
        )
    return loss_name


def uses_weighted_reconstruction_loss(args) -> bool:
    arch = str(args.arch).lower()
    return arch == "convlstm" or (is_predrnnpp_arch(args) and get_predrnnpp_recipe(args) == "simvp")


def uses_local_reconstruction_loss(args) -> bool:
    return (
        not uses_simvp_moganet_openstl_loss(args)
        and not uses_earthfarseer_openstl_loss(args)
        and not uses_predformer_facts_openstl_loss(args)
        and
        not uses_weighted_reconstruction_loss(args)
        and not uses_predrnnpp_openstl_recipe(args)
        and not is_tau_arch(args)
    )


def should_enable_ddp_find_unused_parameters(args) -> bool:
    # OpenSTL's PredRNN++ reference cell registers conv_o but does not consume it in forward().
    # DDP therefore needs unused-parameter detection for PredRNN++ training on multiple GPUs.
    return is_predrnnpp_arch(args)


def resolve_recipe_tag(args) -> str:
    if is_predrnnpp_arch(args):
        return get_predrnnpp_recipe(args)
    if str(args.arch).lower() == "simvp":
        return get_effective_simvp_recipe_from_args(args)
    return "default"


def build_parser():
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
    parser.add_argument("--simvp_model_type", type=str, default="incepu", choices=SIMVP_MODEL_TYPE_ALIASES)
    parser.add_argument("--simvp_recipe", type=str, default="auto", choices=SIMVP_RECIPE_CHOICES)
    parser.add_argument("--simvp_spatio_kernel_enc", type=int, default=3)
    parser.add_argument("--simvp_spatio_kernel_dec", type=int, default=3)
    parser.add_argument("--simvp_mlp_ratio", type=float, default=8.0)
    parser.add_argument("--simvp_drop", type=float, default=0.0)
    parser.add_argument("--simvp_drop_path", type=float, default=0.0)
    parser.add_argument("--convlstm_hidden", type=str, default="128,128,128,128")
    parser.add_argument("--convlstm_filter_size", type=int, default=5)
    parser.add_argument("--convlstm_patch_size", type=int, default=4)
    parser.add_argument("--convlstm_stride", type=int, default=1)
    parser.add_argument("--convlstm_layer_norm", action="store_true")
    parser.add_argument("--predrnnpp_hidden", type=str, default="128,128,128,128")
    parser.add_argument("--predrnnpp_filter_size", type=int, default=5)
    parser.add_argument("--predrnnpp_patch_size", type=int, default=4)
    parser.add_argument("--predrnnpp_stride", type=int, default=1)
    parser.add_argument("--predrnnpp_layer_norm", action="store_true")
    parser.add_argument("--predrnnpp_recipe", type=str, default="simvp", choices=PREDRNNPP_RECIPE_CHOICES)
    parser.add_argument("--reverse_scheduled_sampling", action="store_true")
    parser.add_argument(
        "--scheduled_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sampling_start_value", type=float, default=1.0)
    parser.add_argument("--sampling_stop_iter", type=int, default=50000)
    parser.add_argument("--r_sampling_step_1", type=int, default=25000)
    parser.add_argument("--r_sampling_step_2", type=int, default=50000)
    parser.add_argument("--r_exp_alpha", type=float, default=5000.0)
    parser.add_argument("--predformer_patch_size", type=int, default=16)
    parser.add_argument("--predformer_dim", type=int, default=256, help="PredFormer_FacTS embedding dim. Odd values are supported.")
    parser.add_argument("--predformer_heads", type=int, default=8)
    parser.add_argument("--predformer_dim_head", type=int, default=32)
    parser.add_argument("--predformer_dropout", type=float, default=0.0)
    parser.add_argument("--predformer_attn_dropout", type=float, default=0.0)
    parser.add_argument("--predformer_drop_path", type=float, default=0.0)
    parser.add_argument("--predformer_scale_dim", type=int, default=4)
    parser.add_argument(
        "--predformer_depth",
        type=int,
        default=4,
        help="Shared transformer stack depth for the FacTS port; the same depth is used for both temporal and spatial branches.",
    )
    parser.add_argument(
        "--predformer_loss",
        type=str,
        default="mae",
        choices=("mae", "mse"),
        help="Training loss for arch='predformer_facts'.",
    )
    parser.add_argument("--in_T", type=int, default=8)
    parser.add_argument("--out_T", type=int, default=2)
    parser.add_argument("--arch", type=str, default="simvp", choices=SUPPORTED_ARCHS)
    parser.add_argument("--tau_spatio_kernel_enc", type=int, default=3)
    parser.add_argument("--tau_spatio_kernel_dec", type=int, default=3)
    parser.add_argument("--tau_mlp_ratio", type=float, default=8.0)
    parser.add_argument("--tau_drop", type=float, default=0.0)
    parser.add_argument("--tau_drop_path", type=float, default=0.0)
    parser.add_argument("--tau_alpha", type=float, default=0.1)
    parser.add_argument("--earthfarseer_incep_ker", type=str, default="3,5,7,11")
    parser.add_argument("--earthfarseer_groups", type=int, default=8)
    parser.add_argument("--earthfarseer_num_interactions", type=int, default=3)
    parser.add_argument("--earthfarseer_patch_size", type=int, default=16)
    parser.add_argument("--earthfarseer_embed_dim", type=int, default=768)
    parser.add_argument(
        "--earthfarseer_depth",
        type=int,
        default=12,
        help=(
            "Legacy EarthFarseer Fourier depth alias. Used for both spatial and temporal Fourier stacks "
            "unless --earthfarseer_spatial_depth or --earthfarseer_temporal_depth is set."
        ),
    )
    parser.add_argument(
        "--earthfarseer_spatial_depth",
        type=int,
        default=None,
        help="EarthFarseer FoTF/global Fourier block depth. Defaults to --earthfarseer_depth.",
    )
    parser.add_argument(
        "--earthfarseer_temporal_depth",
        type=int,
        default=None,
        help="EarthFarseer temporal Fourier block depth. Defaults to --earthfarseer_depth.",
    )
    parser.add_argument("--earthfarseer_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--earthfarseer_drop", type=float, default=0.0)
    parser.add_argument("--earthfarseer_drop_path", type=float, default=0.0)
    parser.add_argument("--hybrid_depth", type=int, default=2)
    parser.add_argument("--hybrid_heads", type=int, default=8)
    parser.add_argument("--hybrid_ffn_ratio", type=float, default=4.0)
    parser.add_argument("--hybrid_attn_dropout", type=float, default=0.1)
    parser.add_argument("--hybrid_ffn_dropout", type=float, default=0.1)
    parser.add_argument("--hybrid_drop_path", type=float, default=0.1)
    parser.add_argument("--use_local_branch", action="store_true")
    parser.add_argument("--lambda_global", type=float, default=1.0)
    parser.add_argument("--lambda_local", type=float, default=0.5)
    parser.add_argument("--loss_mae_weight", type=float, default=0.15)
    parser.add_argument("--loss_mse_weight", type=float, default=0.80)
    parser.add_argument("--loss_percep_weight", type=float, default=0.05)
    parser.add_argument(
        "--disable_perceptual_when_untrained_vgg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--opt", type=str, default="auto", choices=["auto", "adam", "adamw"])
    parser.add_argument("--sched", type=str, default="auto", choices=["auto", "none", "cosine", "onecycle"])
    parser.add_argument("--warmup_epoch", type=int, default=0)
    parser.add_argument("--perceptual_vgg_weights", type=str, default="")
    parser.add_argument("--local_top", type=int, default=186)
    parser.add_argument("--local_bottom", type=int, default=410)
    parser.add_argument("--report_local_metrics", action="store_true")
    parser.add_argument(
        "--best_metric_mode",
        type=str,
        default="auto",
        choices=["auto", "global", "local", "combined", "clarity"],
    )
    parser.add_argument("--best_metric_local_weight", type=float, default=1.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./work_dirs/simvp_spreadf")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def collect_explicit_cli_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for token in tokens:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest and dest != "help":
            explicit.add(dest)
    return explicit


def apply_simvp_recipe_defaults(args, explicit_cli_args=None):
    explicit_cli_args = set() if explicit_cli_args is None else set(explicit_cli_args)
    args.simvp_model_type = get_simvp_model_type(args)
    args.simvp_recipe = get_simvp_recipe(args)
    args.predrnnpp_recipe = get_predrnnpp_recipe(args)

    if not uses_simvp_openstl_recipe(args):
        return args

    for field, value in SIMVP_OPENSTL_TRAIN_PRESET.items():
        if field in ("opt", "sched"):
            continue
        if field not in explicit_cli_args:
            setattr(args, field, value)
    return args


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
    x_local = None
    y_local = None
    if "x_local" in batch[0]:
        x_local = torch.stack([item["x_local"] for item in batch], dim=0)
    if "y_local" in batch[0]:
        y_local = torch.stack([item["y_local"] for item in batch], dim=0)
    return x, y, x_local, y_local


def crop_local_region(seq, top: int, bottom: int):
    return seq[:, :, :, top:bottom, :]


def validate_patch_grid(channels: int, height: int, width: int, patch_size: int):
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Input size {(height, width)} must be divisible by patch_size={patch_size}.")
    patch_h = height // patch_size
    patch_w = width // patch_size
    patch_channels = channels * patch_size * patch_size
    return patch_h, patch_w, patch_channels


def build_predrnnpp_real_input_flag(args, batch_size: int, channels: int, height: int, width: int, device, eta: float, itr: int):
    patch_h, patch_w, patch_channels = validate_patch_grid(
        channels=channels,
        height=height,
        width=width,
        patch_size=int(args.predrnnpp_patch_size),
    )

    def _expand(binary_flags: torch.Tensor) -> torch.Tensor:
        if binary_flags.numel() == 0:
            return torch.zeros(batch_size, 0, patch_h, patch_w, patch_channels, device=device)
        return (
            binary_flags.to(dtype=torch.float32)
            .view(binary_flags.shape[0], binary_flags.shape[1], 1, 1, 1)
            .expand(binary_flags.shape[0], binary_flags.shape[1], patch_h, patch_w, patch_channels)
            .contiguous()
        )

    if args.reverse_scheduled_sampling:
        if itr < args.r_sampling_step_1:
            r_eta = 0.5
        elif itr < args.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / float(args.r_exp_alpha))
        else:
            r_eta = 1.0

        if itr < args.r_sampling_step_1:
            future_eta = 0.5
        elif itr < args.r_sampling_step_2:
            denom = max(args.r_sampling_step_2 - args.r_sampling_step_1, 1)
            future_eta = 0.5 - (0.5 / denom) * (itr - args.r_sampling_step_1)
        else:
            future_eta = 0.0

        history_flags = torch.rand(batch_size, max(args.in_T - 1, 0), device=device) < r_eta
        future_flags = torch.rand(batch_size, max(args.out_T - 1, 0), device=device) < future_eta
        history_mask = _expand(history_flags)
        future_mask = _expand(future_flags)
        return eta, torch.cat([history_mask, future_mask], dim=1)

    zeros = torch.zeros(batch_size, max(args.out_T - 1, 0), patch_h, patch_w, patch_channels, device=device)
    if not args.scheduled_sampling or args.out_T <= 1:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        decay = float(args.sampling_start_value) / max(float(args.sampling_stop_iter), 1.0)
        eta = max(0.0, eta - decay)
    else:
        eta = 0.0

    future_flags = torch.rand(batch_size, max(args.out_T - 1, 0), device=device) < eta
    return eta, _expand(future_flags)


def compute_best_score(
    val_mae: float,
    val_local_mae: float,
    val_mse: float,
    val_local_mse: float,
    val_ssim: float,
    val_perceptual: float,
    mode: str,
    local_weight: float,
) -> float:
    if mode == "global":
        return float(val_mae)
    if mode == "local":
        return float(val_local_mae)
    if mode == "clarity":
        # Lower is better. Favor structural sharpness first, then pixel fidelity.
        return float(
            0.55 * max(0.0, 1.0 - val_ssim)
            + 0.25 * val_mse
            + 0.15 * val_local_mse
            + 0.05 * val_perceptual
        )
    return float(val_mae + local_weight * val_local_mae)


def get_amp_autocast(device_type: str, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    if device_type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def create_grad_scaler(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    if device_type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    class _IdentityScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

        def get_scale(self):
            return 1.0

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return None

    return _IdentityScaler()


def step_optimizer_and_maybe_step_scheduler(
    optimizer,
    scaler,
    scheduler,
    scheduler_step_mode: str,
    amp_enabled: bool,
):
    optimizer_step_executed = True

    if amp_enabled:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        # GradScaler lowers the scale when inf/NaN gradients force the optimizer step to be skipped.
        optimizer_step_executed = new_scale >= old_scale
    else:
        optimizer.step()

    # Iter-level schedulers such as OneCycleLR must only advance after a real optimizer update.
    if scheduler is not None and scheduler_step_mode == "iter" and optimizer_step_executed:
        scheduler.step()

    return optimizer_step_executed


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize_hw: Tuple[int, int] = (224, 224), local_weights_path: str = ""):
        super().__init__()
        self.resize_hw = resize_hw
        self.weight_source = "imagenet"
        self.has_pretrained_weights = True

        local_weights_path = str(local_weights_path).strip()
        if local_weights_path:
            backbone = vgg16(weights=None)
            state_dict = torch.load(local_weights_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            backbone.load_state_dict(state_dict, strict=True)
            self.weight_source = f"local:{local_weights_path}"
            features = backbone.features
        else:
            try:
                features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            except Exception:
                features = vgg16(weights=None).features
                self.weight_source = "random"
                self.has_pretrained_weights = False

        self.blocks = nn.ModuleList(
            [
                features[:4].eval(),
                features[4:9].eval(),
                features[9:16].eval(),
            ]
        )
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"VGGPerceptualLoss expects [B, T, C, H, W], but got {x.shape}.")

        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        elif channels != 3:
            raise ValueError(f"VGGPerceptualLoss only supports 1 or 3 channels, but got {channels}.")

        if (height, width) != self.resize_hw:
            x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)

        x = x.float()
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self._prepare(pred)
        target_features = self._prepare(target)

        loss = pred_features.new_zeros(())
        for block in self.blocks:
            pred_features = block(pred_features)
            target_features = block(target_features)
            loss = loss + F.l1_loss(pred_features, target_features)
        return loss / len(self.blocks)


def resolve_best_metric_mode(args):
    if args.best_metric_mode == "auto":
        return "clarity" if uses_weighted_reconstruction_loss(args) else "combined"
    return args.best_metric_mode


def resolve_optimizer_config(args):
    args.predrnnpp_recipe = get_predrnnpp_recipe(args)
    args.simvp_model_type = get_simvp_model_type(args)
    args.simvp_recipe = get_simvp_recipe(args)
    opt = str(args.opt).lower()
    if opt == "auto":
        opt = (
            "adam"
            if uses_predrnnpp_openstl_recipe(args)
            or uses_simvp_openstl_recipe(args)
            or is_tau_arch(args)
            or uses_predformer_facts_openstl_loss(args)
            else "adamw"
        )
    args.opt = opt
    return args


def resolve_scheduler_config(args):
    args.predrnnpp_recipe = get_predrnnpp_recipe(args)
    args.simvp_model_type = get_simvp_model_type(args)
    args.simvp_recipe = get_simvp_recipe(args)
    if args.sched == "auto":
        if uses_predrnnpp_openstl_recipe(args):
            args.sched = "onecycle"
        elif uses_simvp_openstl_recipe(args):
            args.sched = "onecycle"
        elif uses_predformer_facts_openstl_loss(args):
            args.sched = "onecycle"
        elif is_tau_arch(args):
            args.sched = "cosine"
        elif uses_weighted_reconstruction_loss(args):
            args.sched = "cosine"
        else:
            args.sched = "none"
    if (
        uses_weighted_reconstruction_loss(args)
        or uses_predrnnpp_openstl_recipe(args)
        or uses_simvp_openstl_recipe(args)
        or uses_predformer_facts_openstl_loss(args)
        or is_tau_arch(args)
    ):
        args.warmup_epoch = 0
    return args


def resolve_weighted_reconstruction_loss_weights(args, perceptual_criterion, logger=None):
    weights = {
        "mae": float(args.loss_mae_weight),
        "mse": float(args.loss_mse_weight),
        "perceptual": float(args.loss_percep_weight),
    }
    for name, value in weights.items():
        if value < 0:
            raise ValueError(
                f"Weighted reconstruction loss weight '{name}' must be non-negative, but got {value}."
            )

    if not uses_weighted_reconstruction_loss(args):
        weights["perceptual"] = 0.0
        return weights

    if weights["perceptual"] > 0:
        has_valid_vgg = perceptual_criterion is not None and perceptual_criterion.has_pretrained_weights
        if not has_valid_vgg and args.disable_perceptual_when_untrained_vgg:
            if logger is not None:
                logger.warning(
                    f"Disabling {args.arch} perceptual loss because no pretrained VGG16 weights were loaded."
                )
            weights["perceptual"] = 0.0

    if weights["mae"] == 0.0 and weights["mse"] == 0.0 and weights["perceptual"] == 0.0:
        raise ValueError(
            f"{args.arch} loss weights are all zero; at least one loss term must be enabled."
        )

    return weights


def resolve_train_loss_mode(args, loss_weights=None):
    if uses_predrnnpp_openstl_recipe(args):
        return "mse_openstl"
    if uses_simvp_moganet_openstl_loss(args):
        return "mae_openstl"
    if uses_earthfarseer_openstl_loss(args):
        return "mse_openstl"
    if uses_predformer_facts_openstl_loss(args):
        return f"{get_predformer_loss(args)}_openstl"
    if is_tau_arch(args):
        if int(args.out_T) <= 2:
            return f"mse + {float(args.tau_alpha):.4f}*diff_div(inactive_when_out_T<=2)"
        return f"mse + {float(args.tau_alpha):.4f}*diff_div"
    if uses_weighted_reconstruction_loss(args):
        if loss_weights is None:
            return "weighted_reconstruction"
        return (
            f"{loss_weights['mae']:.4f}*MAE + "
            f"{loss_weights['mse']:.4f}*MSE + "
            f"{loss_weights['perceptual']:.4f}*perceptual"
        )
    return "lambda_global*global_l1 + lambda_local*local_l1"


def build_optimizer(args, model):
    # TODO: OpenSTL-style no_weight_decay keywords exposed by MogaSubBlock.no_weight_decay()
    # are not yet wired into optimizer param groups here. The current training path still applies
    # one global weight_decay to model.parameters() for both Adam and AdamW.
    if args.opt == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{args.opt}'.")


def build_lr_scheduler(args, optimizer, steps_per_epoch: int):
    sched = str(args.sched).lower()
    if sched == "none":
        return None, "epoch"
    if sched == "cosine":
        if args.warmup_epoch != 0:
            raise ValueError("warmup_epoch is not implemented in simvp_spreadf train.py; use 0.")
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(args.epochs), 1),
            ),
            "epoch",
        )
    if sched == "onecycle":
        if args.warmup_epoch != 0:
            raise ValueError("warmup_epoch is not used with OneCycleLR; set --warmup_epoch 0.")
        if steps_per_epoch < 1:
            raise ValueError(f"OneCycleLR requires steps_per_epoch >= 1, but got {steps_per_epoch}.")
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                epochs=max(int(args.epochs), 1),
                steps_per_epoch=int(steps_per_epoch),
            ),
            "iter",
        )
    raise ValueError(f"Unsupported scheduler '{args.sched}'.")


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
                "train_loss",
                "train_loss_global",
                "train_loss_local",
                "train_mae",
                "train_mse",
                "train_perceptual",
                "val_mae",
                "val_mse",
                "val_perceptual",
                "val_psnr",
                "val_ssim",
                "val_local_mae",
                "val_local_mse",
                "best_score",
                "best_metric_mode",
                "lr",
                "lr_next",
                "sched",
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
            f'{row["train_loss"]:.8f}',
            f'{row["train_loss_global"]:.8f}',
            f'{row["train_loss_local"]:.8f}',
            f'{row["train_mae"]:.8f}',
            f'{row["train_mse"]:.8f}',
            f'{row["train_perceptual"]:.8f}',
            f'{row["val_mae"]:.8f}',
            f'{row["val_mse"]:.8f}',
            f'{row["val_perceptual"]:.8f}',
            f'{row["val_psnr"]:.8f}',
            f'{row["val_ssim"]:.8f}',
            f'{row["val_local_mae"]:.8f}',
            f'{row["val_local_mse"]:.8f}',
            f'{row["best_score"]:.8f}',
            row["best_metric_mode"],
            f'{row["lr"]:.10f}',
            f'{row["lr_next"]:.10f}',
            row["sched"],
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


def _resolve_ssim_window_size(height: int, width: int, preferred: int = 11) -> int:
    window_size = min(preferred, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    return max(window_size, 1)


def tensor_ssim(pred, target, data_range=1.0, window_size: int = 11, eps: float = 1e-8):
    if pred.shape != target.shape:
        raise ValueError(f"tensor_ssim expects matching shapes, got {pred.shape} and {target.shape}.")
    if pred.ndim != 4:
        raise ValueError(f"tensor_ssim expects [N, C, H, W], but got {pred.shape}.")

    _, _, height, width = pred.shape
    window_size = _resolve_ssim_window_size(height, width, preferred=window_size)
    padding = window_size // 2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    pred = pred.float()
    target = target.float()

    mu_pred = torch.nn.functional.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=padding)
    mu_target = torch.nn.functional.avg_pool2d(target, kernel_size=window_size, stride=1, padding=padding)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = (
        torch.nn.functional.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=padding)
        - mu_pred_sq
    )
    sigma_target_sq = (
        torch.nn.functional.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=padding)
        - mu_target_sq
    )
    sigma_pred_target = (
        torch.nn.functional.avg_pool2d(pred * target, kernel_size=window_size, stride=1, padding=padding)
        - mu_pred_target
    )

    numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=eps)
    return ssim_map.flatten(1).mean(dim=1)


def batch_ssim_sum(pred, target, data_range=1.0):
    """
    返回当前 batch 的 SSIM 累加和，便于跨卡 all_reduce 后再除总样本数。
    pred, target: [B, T, C, H, W]
    """
    batch_size, num_frames, channels, height, width = pred.shape
    pred_frames = pred.detach().reshape(batch_size * num_frames, channels, height, width)
    target_frames = target.detach().reshape(batch_size * num_frames, channels, height, width)
    frame_scores = tensor_ssim(pred_frames, target_frames, data_range=data_range)
    seq_scores = frame_scores.reshape(batch_size, num_frames).mean(dim=1)
    return float(seq_scores.sum().item())


@torch.no_grad()
def evaluate(
    model,
    loader,
    mae_criterion,
    device,
    amp_enabled=False,
    local_top: int = 186,
    local_bottom: int = 410,
    strict_local: bool = False,
    perceptual_criterion: Optional[nn.Module] = None,
):
    model.eval()

    total_mae = 0.0
    total_mse = 0.0
    total_perceptual = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_count = 0.0
    total_local_mae = 0.0
    total_local_mse = 0.0
    total_local_count = 0.0

    iterator = tqdm(loader, desc="val", leave=False) if is_main_process() else loader

    for x, y, x_local, y_local in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if x_local is not None:
            x_local = x_local.to(device, non_blocking=True)
        if y_local is not None:
            y_local = y_local.to(device, non_blocking=True)

        with get_amp_autocast(device.type, amp_enabled):
            pred = model(x, x_local=x_local, strict_local=strict_local)
            mae = mae_criterion(pred, y)
            if y_local is not None:
                pred_local = crop_local_region(pred, local_top, local_bottom)
                local_mae = mae_criterion(pred_local, y_local)
            else:
                local_mae = pred.new_zeros(())

        mse = tensor_mse(pred, y).item()
        if perceptual_criterion is not None:
            perceptual = float(perceptual_criterion(pred.float(), y.float()).item())
        else:
            perceptual = 0.0
        psnr = tensor_psnr(pred, y).item()
        ssim_sum = batch_ssim_sum(pred, y)
        bs = x.size(0)

        total_mae += mae.item() * bs
        total_mse += mse * bs
        total_perceptual += perceptual * bs
        total_psnr += psnr * bs
        total_ssim += ssim_sum
        total_count += bs
        if y_local is not None:
            local_mse = tensor_mse(pred_local, y_local).item()
            total_local_mae += local_mae.item() * bs
            total_local_mse += local_mse * bs
            total_local_count += bs

    total_mae = reduce_sum_scalar(total_mae, device)
    total_mse = reduce_sum_scalar(total_mse, device)
    total_perceptual = reduce_sum_scalar(total_perceptual, device)
    total_psnr = reduce_sum_scalar(total_psnr, device)
    total_ssim = reduce_sum_scalar(total_ssim, device)
    total_count = reduce_sum_scalar(total_count, device)
    total_local_mae = reduce_sum_scalar(total_local_mae, device)
    total_local_mse = reduce_sum_scalar(total_local_mse, device)
    total_local_count = reduce_sum_scalar(total_local_count, device)

    metrics = {
        "val_mae": total_mae / max(total_count, 1.0),
        "val_mse": total_mse / max(total_count, 1.0),
        "val_perceptual": total_perceptual / max(total_count, 1.0),
        "val_psnr": total_psnr / max(total_count, 1.0),
        "val_ssim": total_ssim / max(total_count, 1.0),
        "val_local_mae": total_local_mae / max(total_local_count, 1.0),
        "val_local_mse": total_local_mse / max(total_local_count, 1.0),
    }
    return metrics


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scaler,
    scheduler,
    args,
    best_epoch,
    best_val_mae,
    history,
    status,
    best_score: Optional[float] = None,
    best_metric_mode: Optional[str] = None,
):
    raw_model = unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "history": history,
        "status": status,
    }
    if best_score is not None:
        ckpt["best_score"] = best_score
    if best_metric_mode is not None:
        ckpt["best_metric_mode"] = best_metric_mode
    torch.save(ckpt, path)


def write_report(
    report_path,
    status,
    reason,
    total_epochs,
    completed_epochs,
    best_epoch,
    best_val_mae,
    history,
    best_score: Optional[float] = None,
    best_metric_mode: Optional[str] = None,
):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SimVP Training Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"status: {status}\n")
        f.write(f"reason: {reason}\n")
        f.write(f"total_epochs: {total_epochs}\n")
        f.write(f"completed_epochs: {completed_epochs}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        if best_metric_mode is not None:
            f.write(f"best_metric_mode: {best_metric_mode}\n")
        if best_score is not None:
            f.write(f"best_score: {best_score:.8f}\n")
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
    parser = build_parser()
    args = parser.parse_args()
    explicit_cli_args = collect_explicit_cli_args(parser)
    args = apply_simvp_recipe_defaults(args, explicit_cli_args=explicit_cli_args)
    args = resolve_optimizer_config(args)
    args = resolve_scheduler_config(args)
    args.best_metric_mode = resolve_best_metric_mode(args)
    args.train_loss_mode = resolve_train_loss_mode(args)

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
        logger.info(f"resolved_recipe: {resolve_recipe_tag(args)}")
        logger.info(f"resolved_optimizer: {args.opt}")
        logger.info(f"resolved_scheduler: {args.sched}")
        logger.info(f"resolved_train_loss_mode: {args.train_loss_mode}")
        logger.info(f"sched: {args.sched}  warmup_epoch: {args.warmup_epoch}")
        logger.info(f"best_metric_mode: {args.best_metric_mode}")
        logger.info(
            f"lambda_global: {args.lambda_global}  lambda_local: {args.lambda_local}  "
            f"local_crop: ({args.local_top}, {args.local_bottom})"
        )
        logger.info(f"report_local_metrics: {args.report_local_metrics}")
        logger.info(f"best_metric_local_weight: {args.best_metric_local_weight}")
        if args.arch == "simvp":
            logger.info(
                f"simvp_model_type: {args.simvp_model_type}  "
                f"simvp_recipe: {args.simvp_recipe}  "
                f"resolved_simvp_recipe: {get_effective_simvp_recipe_from_args(args)}"
            )
            if uses_simvp_openstl_recipe(args):
                logger.info(
                    f"simvp_spatio_kernel_enc: {args.simvp_spatio_kernel_enc}  "
                    f"simvp_spatio_kernel_dec: {args.simvp_spatio_kernel_dec}  "
                    f"simvp_mlp_ratio: {args.simvp_mlp_ratio}  "
                    f"simvp_drop: {args.simvp_drop}  "
                    f"simvp_drop_path: {args.simvp_drop_path}"
                )
        if is_tau_arch(args):
            logger.info(
                f"tau_spatio_kernel_enc: {args.tau_spatio_kernel_enc}  "
                f"tau_spatio_kernel_dec: {args.tau_spatio_kernel_dec}  "
                f"tau_mlp_ratio: {args.tau_mlp_ratio}  "
                f"tau_drop: {args.tau_drop}  "
                f"tau_drop_path: {args.tau_drop_path}  "
                f"tau_alpha: {args.tau_alpha}"
            )
            if args.out_T <= 2:
                logger.warning(
                    "TAU diff_div_reg is inactive when out_T<=2. Under the current dataset setting, "
                    "the TAU training loss reduces to plain MSE."
                )
        if args.arch == "earthfarseer":
            logger.info(
                f"earthfarseer_incep_ker: {args.earthfarseer_incep_ker}  "
                f"earthfarseer_groups: {args.earthfarseer_groups}  "
                f"earthfarseer_num_interactions: {args.earthfarseer_num_interactions}  "
                f"earthfarseer_patch_size: {args.earthfarseer_patch_size}  "
                f"earthfarseer_embed_dim: {args.earthfarseer_embed_dim}  "
                f"earthfarseer_depth: {args.earthfarseer_depth}  "
                f"earthfarseer_spatial_depth: {args.earthfarseer_spatial_depth}  "
                f"earthfarseer_temporal_depth: {args.earthfarseer_temporal_depth}  "
                f"earthfarseer_mlp_ratio: {args.earthfarseer_mlp_ratio}  "
                f"earthfarseer_drop: {args.earthfarseer_drop}  "
                f"earthfarseer_drop_path: {args.earthfarseer_drop_path}"
            )
        if args.arch == "hybrid_unet_facts":
            logger.info(
                "hybrid_unet_facts uses a strict Fac-T-S translator, a cross-attention bottleneck forecaster, "
                "and lightweight temporal-conv skip forecasters."
            )
            logger.info(f"use_local_branch: {args.use_local_branch}")
        if args.arch == "convlstm":
            logger.info(
                f"convlstm_hidden: {args.convlstm_hidden}  "
                f"filter_size: {args.convlstm_filter_size}  "
                f"patch_size: {args.convlstm_patch_size}  "
                f"stride: {args.convlstm_stride}  "
                f"layer_norm: {args.convlstm_layer_norm}"
            )
        if args.arch == "predrnnpp":
            logger.info(
                f"predrnnpp_hidden: {args.predrnnpp_hidden}  "
                f"filter_size: {args.predrnnpp_filter_size}  "
                f"patch_size: {args.predrnnpp_patch_size}  "
                f"stride: {args.predrnnpp_stride}  "
                f"layer_norm: {args.predrnnpp_layer_norm}  "
                f"recipe: {args.predrnnpp_recipe}  "
                f"reverse_scheduled_sampling: {args.reverse_scheduled_sampling}  "
                f"scheduled_sampling: {args.scheduled_sampling}"
            )
        if args.arch == "predformer_facts":
            logger.info(
                f"predformer_patch_size: {args.predformer_patch_size}  "
                f"predformer_dim: {args.predformer_dim}  "
                f"predformer_heads: {args.predformer_heads}  "
                f"predformer_dim_head: {args.predformer_dim_head}  "
                f"predformer_dropout: {args.predformer_dropout}  "
                f"predformer_attn_dropout: {args.predformer_attn_dropout}  "
                f"predformer_drop_path: {args.predformer_drop_path}  "
                f"predformer_scale_dim: {args.predformer_scale_dim}  "
                f"predformer_depth: {args.predformer_depth}  "
                f"predformer_loss: {args.predformer_loss}"
            )

    # Local targets are always kept available because local metrics and best-model
    # selection can depend on them even when local logging is disabled.
    local_crop = (args.local_top, args.local_bottom)
    train_set = IonogramManifestDataset(
        manifest_path=args.train_manifest,
        image_mode=args.image_mode,
        image_size=args.image_size,
        local_crop=local_crop,
    )
    val_set = IonogramManifestDataset(
        manifest_path=args.val_manifest,
        image_mode=args.image_mode,
        image_size=args.image_size,
        local_crop=local_crop,
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
        simvp_model_type=args.simvp_model_type,
        simvp_recipe=args.simvp_recipe,
        simvp_spatio_kernel_enc=args.simvp_spatio_kernel_enc,
        simvp_spatio_kernel_dec=args.simvp_spatio_kernel_dec,
        simvp_mlp_ratio=args.simvp_mlp_ratio,
        simvp_drop=args.simvp_drop,
        simvp_drop_path=args.simvp_drop_path,
        tau_spatio_kernel_enc=args.tau_spatio_kernel_enc,
        tau_spatio_kernel_dec=args.tau_spatio_kernel_dec,
        tau_mlp_ratio=args.tau_mlp_ratio,
        tau_drop=args.tau_drop,
        tau_drop_path=args.tau_drop_path,
        earthfarseer_incep_ker=args.earthfarseer_incep_ker,
        earthfarseer_groups=args.earthfarseer_groups,
        earthfarseer_num_interactions=args.earthfarseer_num_interactions,
        earthfarseer_patch_size=args.earthfarseer_patch_size,
        earthfarseer_embed_dim=args.earthfarseer_embed_dim,
        earthfarseer_depth=args.earthfarseer_depth,
        earthfarseer_spatial_depth=args.earthfarseer_spatial_depth,
        earthfarseer_temporal_depth=args.earthfarseer_temporal_depth,
        earthfarseer_mlp_ratio=args.earthfarseer_mlp_ratio,
        earthfarseer_drop=args.earthfarseer_drop,
        earthfarseer_drop_path=args.earthfarseer_drop_path,
        convlstm_hidden=args.convlstm_hidden,
        convlstm_filter_size=args.convlstm_filter_size,
        convlstm_patch_size=args.convlstm_patch_size,
        convlstm_stride=args.convlstm_stride,
        convlstm_layer_norm=args.convlstm_layer_norm,
        predrnnpp_hidden=args.predrnnpp_hidden,
        predrnnpp_filter_size=args.predrnnpp_filter_size,
        predrnnpp_patch_size=args.predrnnpp_patch_size,
        predrnnpp_stride=args.predrnnpp_stride,
        predrnnpp_layer_norm=args.predrnnpp_layer_norm,
        predrnnpp_recipe=args.predrnnpp_recipe,
        predrnnpp_reverse_scheduled_sampling=args.reverse_scheduled_sampling,
        predformer_patch_size=args.predformer_patch_size,
        predformer_dim=args.predformer_dim,
        predformer_heads=args.predformer_heads,
        predformer_dim_head=args.predformer_dim_head,
        predformer_dropout=args.predformer_dropout,
        predformer_attn_dropout=args.predformer_attn_dropout,
        predformer_drop_path=args.predformer_drop_path,
        predformer_scale_dim=args.predformer_scale_dim,
        predformer_depth=args.predformer_depth,
        arch=args.arch,
        hybrid_depth=args.hybrid_depth,
        hybrid_heads=args.hybrid_heads,
        hybrid_ffn_ratio=args.hybrid_ffn_ratio,
        hybrid_attn_dropout=args.hybrid_attn_dropout,
        hybrid_ffn_dropout=args.hybrid_ffn_dropout,
        hybrid_drop_path=args.hybrid_drop_path,
        use_local_branch=args.use_local_branch,
        local_crop=(args.local_top, args.local_bottom),
    ).to(device)

    ddp_find_unused_parameters = should_enable_ddp_find_unused_parameters(args)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
        )

    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    perceptual_criterion = None
    if uses_weighted_reconstruction_loss(args):
        perceptual_criterion = VGGPerceptualLoss(
            local_weights_path=args.perceptual_vgg_weights,
        ).to(device)
        perceptual_criterion.eval()
        if is_main_process():
            if perceptual_criterion.weight_source == "random":
                logger.warning(
                    f"{args.arch} perceptual loss backend: VGG16 random weights."
                )
            else:
                logger.info(
                    f"{args.arch} perceptual loss backend: VGG16 ({perceptual_criterion.weight_source} weights)"
                )
    loss_weights = resolve_weighted_reconstruction_loss_weights(
        args=args,
        perceptual_criterion=perceptual_criterion,
        logger=logger if is_main_process() else None,
    )
    if uses_weighted_reconstruction_loss(args):
        args.loss_mae_weight_effective = loss_weights["mae"]
        args.loss_mse_weight_effective = loss_weights["mse"]
        args.loss_percep_weight_effective = loss_weights["perceptual"]
    else:
        args.loss_mae_weight_effective = 0.0
        args.loss_mse_weight_effective = 0.0
        args.loss_percep_weight_effective = 0.0
    args.train_loss_mode = resolve_train_loss_mode(args, loss_weights=loss_weights)
    if is_main_process():
        logger.info(f"resolved_train_loss_mode: {args.train_loss_mode}")
    optimizer = build_optimizer(args, model)
    scheduler, scheduler_step_mode = build_lr_scheduler(args, optimizer, steps_per_epoch=len(train_loader))
    if is_main_process():
        logger.info(f"ddp_find_unused_parameters: {ddp_find_unused_parameters}")
        logger.info(f"optimizer_impl: {type(optimizer).__name__}")
        logger.info(f"scheduler_impl: {type(scheduler).__name__ if scheduler is not None else 'None'}")
    scaler = create_grad_scaler(device.type, amp_enabled)

    best_epoch = 0
    best_val_mae = float("inf")
    best_score = float("inf")
    history = []

    status = "RUNNING"
    reason = ""
    completed_epochs = 0
    global_step = 0
    predrnnpp_sampling_eta = float(args.sampling_start_value)

    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            model.train()
            epoch_start_lr = optimizer.param_groups[0]["lr"]

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_loss_sum = 0.0
            train_loss_global_sum = 0.0
            train_loss_local_sum = 0.0
            train_mse_sum = 0.0
            train_perceptual_sum = 0.0
            train_diff_div_sum = 0.0
            train_count = 0.0
            train_local_count = 0.0

            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}") if is_main_process() else train_loader

            for x, y, x_local, y_local in iterator:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if x_local is not None:
                    x_local = x_local.to(device, non_blocking=True)
                if y_local is not None:
                    y_local = y_local.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with get_amp_autocast(device.type, amp_enabled):
                    if uses_predrnnpp_openstl_recipe(args):
                        full_sequence = torch.cat([x, y], dim=1)
                        predrnnpp_sampling_eta, mask_true = build_predrnnpp_real_input_flag(
                            args=args,
                            batch_size=x.size(0),
                            channels=x.shape[2],
                            height=x.shape[3],
                            width=x.shape[4],
                            device=device,
                            eta=predrnnpp_sampling_eta,
                            itr=global_step,
                        )
                        pred, loss = model(
                            full_sequence,
                            mask_true=mask_true,
                            return_loss=True,
                        )
                        loss_local = pred.new_zeros(())
                    else:
                        pred = model(x, x_local=x_local, strict_local=args.use_local_branch)
                        if uses_local_reconstruction_loss(args):
                            if y_local is not None:
                                pred_local = crop_local_region(pred, args.local_top, args.local_bottom)
                                loss_local = mae_criterion(pred_local, y_local)
                            else:
                                loss_local = pred.new_zeros(())
                            loss = args.lambda_global * mae_criterion(pred, y) + args.lambda_local * loss_local
                        else:
                            loss_local = pred.new_zeros(())

                    loss_mae = mae_criterion(pred, y)
                    loss_mse = mse_criterion(pred, y)

                if uses_weighted_reconstruction_loss(args):
                    if loss_weights["perceptual"] > 0.0:
                        loss_perceptual = perceptual_criterion(pred.float(), y.float())
                    else:
                        loss_perceptual = pred.new_zeros(())
                    loss_diff_div = pred.new_zeros(())
                    loss = (
                        loss_weights["mae"] * loss_mae
                        + loss_weights["mse"] * loss_mse
                        + loss_weights["perceptual"] * loss_perceptual
                    )
                elif is_tau_arch(args):
                    loss_perceptual = pred.new_zeros(())
                    loss_diff_div = tau_diff_div_reg(
                        pred.float(),
                        y.float(),
                    )
                    loss = loss_mse + float(args.tau_alpha) * loss_diff_div
                elif (
                    uses_earthfarseer_openstl_loss(args)
                ):
                    loss_perceptual = pred.new_zeros(())
                    loss_diff_div = pred.new_zeros(())
                    loss = loss_mse
                elif uses_predformer_facts_openstl_loss(args):
                    loss_perceptual = pred.new_zeros(())
                    loss_diff_div = pred.new_zeros(())
                    loss = loss_mae if get_predformer_loss(args) == "mae" else loss_mse
                elif uses_simvp_moganet_openstl_loss(args):
                    loss_perceptual = pred.new_zeros(())
                    loss_diff_div = pred.new_zeros(())
                    loss = loss_mae
                else:
                    loss_perceptual = pred.new_zeros(())
                    loss_diff_div = pred.new_zeros(())

                scaler.scale(loss).backward()
                # Keep the optimizer/scaler update ahead of the iter-level scheduler update.
                # Under AMP, skip advancing OneCycleLR when GradScaler suppresses the optimizer step.
                step_optimizer_and_maybe_step_scheduler(
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    scheduler_step_mode=scheduler_step_mode,
                    amp_enabled=amp_enabled,
                )
                global_step += 1

                bs = x.size(0)
                train_loss_sum += loss.item() * bs
                train_loss_global_sum += loss_mae.item() * bs
                train_mse_sum += loss_mse.item() * bs
                train_perceptual_sum += loss_perceptual.item() * bs
                train_diff_div_sum += loss_diff_div.item() * bs
                train_count += bs
                if uses_local_reconstruction_loss(args) and y_local is not None:
                    train_loss_local_sum += loss_local.item() * bs
                    train_local_count += bs

                if is_main_process():
                    if uses_weighted_reconstruction_loss(args):
                        iterator.set_postfix(
                            loss=f"{loss.item():.6f}",
                            mae=f"{loss_mae.item():.6f}",
                            mse=f"{loss_mse.item():.6f}",
                            perceptual=f"{loss_perceptual.item():.6f}",
                        )
                    elif uses_predrnnpp_openstl_recipe(args):
                        iterator.set_postfix(
                            loss=f"{loss.item():.6f}",
                            mae=f"{loss_mae.item():.6f}",
                            mse=f"{loss_mse.item():.6f}",
                        )
                    elif is_tau_arch(args):
                        iterator.set_postfix(
                            loss=f"{loss.item():.6f}",
                            mse=f"{loss_mse.item():.6f}",
                            diff_div=f"{loss_diff_div.item():.6f}",
                        )
                    elif (
                        uses_earthfarseer_openstl_loss(args)
                    ):
                        iterator.set_postfix(
                            loss=f"{loss.item():.6f}",
                            mse=f"{loss_mse.item():.6f}",
                        )
                    elif uses_predformer_facts_openstl_loss(args):
                        if get_predformer_loss(args) == "mae":
                            iterator.set_postfix(
                                loss=f"{loss.item():.6f}",
                                mae=f"{loss_mae.item():.6f}",
                            )
                        else:
                            iterator.set_postfix(
                                loss=f"{loss.item():.6f}",
                                mse=f"{loss_mse.item():.6f}",
                            )
                    elif uses_simvp_moganet_openstl_loss(args):
                        iterator.set_postfix(
                            loss=f"{loss.item():.6f}",
                            mae=f"{loss_mae.item():.6f}",
                        )
                    else:
                        iterator.set_postfix(
                            loss=f"{loss.item():.6f}",
                            global_l1=f"{loss_mae.item():.6f}",
                            local_l1=f"{loss_local.item():.6f}",
                        )

            train_loss_sum = reduce_sum_scalar(train_loss_sum, device)
            train_loss_global_sum = reduce_sum_scalar(train_loss_global_sum, device)
            train_loss_local_sum = reduce_sum_scalar(train_loss_local_sum, device)
            train_mse_sum = reduce_sum_scalar(train_mse_sum, device)
            train_perceptual_sum = reduce_sum_scalar(train_perceptual_sum, device)
            train_diff_div_sum = reduce_sum_scalar(train_diff_div_sum, device)
            train_count = reduce_sum_scalar(train_count, device)
            train_local_count = reduce_sum_scalar(train_local_count, device)
            train_loss = train_loss_sum / max(train_count, 1.0)
            train_loss_global = train_loss_global_sum / max(train_count, 1.0)
            train_loss_local = train_loss_local_sum / max(train_local_count, 1.0)
            train_mse = train_mse_sum / max(train_count, 1.0)
            train_perceptual = train_perceptual_sum / max(train_count, 1.0)
            train_diff_div = train_diff_div_sum / max(train_count, 1.0)
            train_mae = train_loss_global

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                mae_criterion=mae_criterion,
                device=device,
                amp_enabled=amp_enabled,
                local_top=args.local_top,
                local_bottom=args.local_bottom,
                strict_local=args.use_local_branch,
                perceptual_criterion=(
                    perceptual_criterion
                    if uses_weighted_reconstruction_loss(args) and loss_weights["perceptual"] > 0.0
                    else None
                ),
            )

            val_mae = val_metrics["val_mae"]
            val_mse = val_metrics["val_mse"]
            val_perceptual = val_metrics["val_perceptual"]
            val_psnr = val_metrics["val_psnr"]
            val_ssim = val_metrics["val_ssim"]
            val_local_mae = val_metrics["val_local_mae"]
            val_local_mse = val_metrics["val_local_mse"]
            epoch_best_score = compute_best_score(
                val_mae=val_mae,
                val_local_mae=val_local_mae,
                val_mse=val_mse,
                val_local_mse=val_local_mse,
                val_ssim=val_ssim,
                val_perceptual=val_perceptual,
                mode=args.best_metric_mode,
                local_weight=args.best_metric_local_weight,
            )

            lr = epoch_start_lr
            if scheduler is not None and scheduler_step_mode == "epoch":
                scheduler.step()
            lr_next = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start_time

            if device.type == "cuda":
                gpu_mem_mb_local = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
            else:
                gpu_mem_mb_local = 0.0

            gpu_mem_mb = reduce_sum_scalar(gpu_mem_mb_local, device) / get_world_size()

            if is_main_process():
                previous_best_score = best_score
                if epoch_best_score < best_score:
                    best_score = epoch_best_score
                    best_val_mae = val_mae
                    best_epoch = epoch
                    is_best = True
                    if math.isfinite(previous_best_score):
                        refresh_reason = (
                            f"best_score improved under mode={args.best_metric_mode}: "
                            f"{epoch_best_score:.6f} < {previous_best_score:.6f}"
                        )
                    else:
                        refresh_reason = (
                            f"best_score initialized under mode={args.best_metric_mode}: "
                            f"{epoch_best_score:.6f}"
                        )
                else:
                    is_best = False
                    refresh_reason = ""

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_loss_global": train_loss_global,
                    "train_loss_local": train_loss_local,
                    "train_mae": train_mae,
                    "train_mse": train_mse,
                    "train_perceptual": train_perceptual,
                    "train_loss_mode": args.train_loss_mode,
                    "sched": args.sched,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "val_perceptual": val_perceptual,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_local_mae": val_local_mae,
                    "val_local_mse": val_local_mse,
                    "best_score": epoch_best_score,
                    "best_metric_mode": args.best_metric_mode,
                    "lr": lr,
                    "lr_next": lr_next,
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
                    scheduler=scheduler,
                    args=args,
                    best_epoch=best_epoch,
                    best_val_mae=best_val_mae,
                    history=history,
                    status="RUNNING",
                    best_score=best_score,
                    best_metric_mode=args.best_metric_mode,
                )

                logger.info(f"[Epoch {epoch}/{args.epochs}]")
                if uses_weighted_reconstruction_loss(args):
                    logger.info(
                        f"train_loss: {train_loss:.4f}  train_mae: {train_mae:.4f}  "
                        f"train_mse: {train_mse:.4f}  train_perceptual: {train_perceptual:.4f}"
                    )
                elif uses_predrnnpp_openstl_recipe(args):
                    logger.info(
                        f"train_loss: {train_loss:.4f}  train_mae: {train_mae:.4f}  "
                        f"train_mse: {train_mse:.4f}  recipe: {args.predrnnpp_recipe}"
                    )
                elif is_tau_arch(args):
                    logger.info(
                        f"train_loss: {train_loss:.4f}  train_mse: {train_mse:.4f}  "
                        f"train_diff_div: {train_diff_div:.4f}  tau_alpha: {args.tau_alpha:.4f}"
                    )
                elif (
                    uses_earthfarseer_openstl_loss(args)
                ):
                    logger.info(
                        f"train_loss: {train_loss:.4f}  train_mse: {train_mse:.4f}  "
                        f"arch: {args.arch}"
                    )
                elif uses_predformer_facts_openstl_loss(args):
                    if get_predformer_loss(args) == "mae":
                        logger.info(
                            f"train_loss: {train_loss:.4f}  train_mae: {train_mae:.4f}  "
                            f"arch: {args.arch}"
                        )
                    else:
                        logger.info(
                            f"train_loss: {train_loss:.4f}  train_mse: {train_mse:.4f}  "
                            f"arch: {args.arch}"
                        )
                elif uses_simvp_moganet_openstl_loss(args):
                    logger.info(
                        f"train_loss: {train_loss:.4f}  train_mae: {train_mae:.4f}  "
                        f"simvp_model_type: {args.simvp_model_type}  simvp_recipe: {args.simvp_recipe}"
                    )
                else:
                    logger.info(
                        f"train_loss: {train_loss:.4f}  train_loss_global: {train_loss_global:.4f}  "
                        f"train_loss_local: {train_loss_local:.4f}"
                    )
                logger.info(
                    f"val_mse: {val_mse:.4f}  val_perceptual: {val_perceptual:.4f}  "
                    f"val_psnr: {val_psnr:.4f}  val_ssim: {val_ssim:.4f}"
                )
                logger.info(f"train_mae(global_l1): {train_mae:.4f}  val_mae: {val_mae:.4f}")
                logger.info(
                    f"best_metric_mode: {args.best_metric_mode}  epoch_best_score: {epoch_best_score:.4f}"
                )
                if args.report_local_metrics:
                    logger.info(
                        f"val_local_mae: {val_local_mae:.4f}  val_local_mse: {val_local_mse:.4f}"
                    )
                logger.info(
                    f"lr_used: {lr:.6f}  lr_next: {lr_next:.6f}  epoch_time: {epoch_time:.1f}s  "
                    f"gpu_mem: {gpu_mem_mb:.0f}MB"
                )
                logger.info(
                    f"best_epoch: {best_epoch}  best_val_mae: {best_val_mae:.4f}  best_score: {best_score:.4f}"
                )

                if is_best:
                    best_path = os.path.join(ckpt_dir, "best.ckpt")
                    save_checkpoint(
                        path=best_path,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        args=args,
                        best_epoch=best_epoch,
                        best_val_mae=best_val_mae,
                        history=history,
                        status="BEST",
                        best_score=best_score,
                        best_metric_mode=args.best_metric_mode,
                    )
                    logger.info(f"{refresh_reason}. Saving best checkpoint to {best_path}")

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
                        scheduler=scheduler if "scheduler" in locals() else None,
                        args=args,
                        best_epoch=best_epoch,
                        best_val_mae=best_val_mae,
                        history=history,
                        status=status,
                        best_score=best_score,
                        best_metric_mode=args.best_metric_mode,
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
                    best_score=best_score if math.isfinite(best_score) else -1.0,
                    best_metric_mode=args.best_metric_mode,
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
            if math.isfinite(best_score):
                logger.info(
                    f"Best score ({args.best_metric_mode}): {best_score:.6f}"
                )

        cleanup_distributed()


if __name__ == "__main__":
    main()
