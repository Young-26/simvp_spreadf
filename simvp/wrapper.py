import torch
import torch.nn as nn
from typing import Optional, Tuple

from .convlstm_model import ConvLSTM_Model
from .earthfarseer_model import EarthFarseer_Model
from .hybrid_unet_facts import HybridUNetFacTS
from .mau_model import MAU_Model
from .mim_model import MIM_Model
from .model import SimVP
from .predformer_facts_model import PredFormerFacTS_Model
from .predformer_quadruplet_tsst_model import PredFormerQuadrupletTSST_Model
from .predrnnpp_model import PredRNNpp_Model
from .predrnnv2_model import PredRNNv2_Model
from .simvp_config import (
    PREDRNNPP_RECIPE_CHOICES,
    get_effective_simvp_recipe,
    normalize_predrnnpp_recipe,
    normalize_simvp_model_type,
    normalize_simvp_recipe,
)
from .tau_model import TAU_Model


SUPPORTED_ARCHS = (
    "simvp",
    "tau",
    "earthfarseer",
    "hybrid_unet_facts",
    "convlstm",
    "mau",
    "mim",
    "predrnnpp",
    "predrnnv2",
    "predformer_facts",
    "predformer_quadruplet_tsst",
)


def _resolve_reverse_scheduled_sampling(
    reverse_scheduled_sampling: Optional[bool],
    *,
    predrnnpp_reverse_scheduled_sampling: Optional[bool] = None,
    predrnnv2_reverse_scheduled_sampling: Optional[bool] = None,
) -> bool:
    """Resolve one effective RSS switch while tolerating legacy alias kwargs."""
    resolved = None if reverse_scheduled_sampling is None else bool(reverse_scheduled_sampling)
    for legacy_name, legacy_value in (
        ("predrnnpp_reverse_scheduled_sampling", predrnnpp_reverse_scheduled_sampling),
        ("predrnnv2_reverse_scheduled_sampling", predrnnv2_reverse_scheduled_sampling),
    ):
        if legacy_value is None:
            continue
        legacy_value = bool(legacy_value)
        if resolved is None:
            resolved = legacy_value
        elif legacy_value != resolved:
            raise ValueError(
                "reverse_scheduled_sampling conflicts with legacy alias "
                f"{legacy_name}={legacy_value}. Use only reverse_scheduled_sampling."
            )
    return False if resolved is None else resolved


class SimVPForecast(nn.Module):
    def __init__(
        self,
        in_T: int = 8,
        out_T: int = 2,
        C: int = 1,
        H: int = 448,
        W: int = 448,
        hid_S: int = 32,
        hid_T: int = 128,
        N_S: int = 4,
        N_T: int = 4,
        simvp_model_type: str = "incepu",
        simvp_recipe: str = "auto",
        simvp_spatio_kernel_enc: int = 3,
        simvp_spatio_kernel_dec: int = 3,
        simvp_mlp_ratio: float = 8.0,
        simvp_drop: float = 0.0,
        simvp_drop_path: float = 0.0,
        convlstm_hidden: str = "128,128,128,128",
        convlstm_filter_size: int = 5,
        convlstm_patch_size: int = 4,
        convlstm_stride: int = 1,
        convlstm_layer_norm: bool = False,
        mim_hidden: str = "128,128,128,128",
        mim_filter_size: int = 5,
        mim_patch_size: int = 4,
        mim_stride: int = 1,
        mim_layer_norm: bool = False,
        mau_hidden: str = "64,64,64,64",
        mau_filter_size: int = 5,
        mau_patch_size: int = 1,
        mau_stride: int = 1,
        mau_sr_size: int = 4,
        mau_tau: int = 5,
        mau_cell_mode: str = "normal",
        mau_model_mode: str = "normal",
        mau_layer_norm: bool = True,
        mau_loss_mode: str = "future_only",
        mau_conv_bias: bool = True,
        predrnnpp_hidden: str = "128,128,128,128",
        predrnnpp_filter_size: int = 5,
        predrnnpp_patch_size: int = 4,
        predrnnpp_stride: int = 1,
        predrnnpp_layer_norm: bool = False,
        predrnnpp_recipe: str = "simvp",
        reverse_scheduled_sampling: Optional[bool] = None,
        predrnnpp_reverse_scheduled_sampling: Optional[bool] = None,
        predrnnv2_hidden: str = "128,128,128,128",
        predrnnv2_filter_size: int = 5,
        predrnnv2_patch_size: int = 4,
        predrnnv2_stride: int = 1,
        predrnnv2_layer_norm: bool = False,
        predrnnv2_decouple_beta: float = 0.1,
        predrnnv2_reverse_scheduled_sampling: Optional[bool] = None,
        predformer_patch_size: int = 16,
        predformer_dim: int = 256,
        predformer_heads: int = 8,
        predformer_dim_head: int = 32,
        predformer_dropout: float = 0.0,
        predformer_attn_dropout: float = 0.0,
        predformer_drop_path: float = 0.0,
        predformer_scale_dim: int = 4,
        predformer_depth: int = 4,
        predformer_transformer_depth: int = 1,
        arch: str = "simvp",
        hybrid_depth: int = 2,
        hybrid_heads: int = 8,
        hybrid_ffn_ratio: float = 4.0,
        hybrid_attn_dropout: float = 0.1,
        hybrid_ffn_dropout: float = 0.1,
        hybrid_drop_path: float = 0.1,
        use_local_branch: bool = False,
        local_crop: Tuple[int, int] = (186, 410),
        tau_spatio_kernel_enc: int = 3,
        tau_spatio_kernel_dec: int = 3,
        tau_mlp_ratio: float = 8.0,
        tau_drop: float = 0.0,
        tau_drop_path: float = 0.0,
        earthfarseer_incep_ker="3,5,7,11",
        earthfarseer_groups: int = 8,
        earthfarseer_num_interactions: int = 3,
        earthfarseer_patch_size: int = 16,
        earthfarseer_embed_dim: int = 768,
        earthfarseer_depth: int = 12,
        earthfarseer_spatial_depth: Optional[int] = None,
        earthfarseer_temporal_depth: Optional[int] = None,
        earthfarseer_mlp_ratio: float = 4.0,
        earthfarseer_drop: float = 0.0,
        earthfarseer_drop_path: float = 0.0,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.out_T = out_T
        self.predrnnpp_recipe = normalize_predrnnpp_recipe(predrnnpp_recipe)
        self.simvp_model_type = normalize_simvp_model_type(simvp_model_type)
        self.simvp_recipe = normalize_simvp_recipe(simvp_recipe)
        self.reverse_scheduled_sampling = _resolve_reverse_scheduled_sampling(
            reverse_scheduled_sampling,
            predrnnpp_reverse_scheduled_sampling=predrnnpp_reverse_scheduled_sampling,
            predrnnv2_reverse_scheduled_sampling=predrnnv2_reverse_scheduled_sampling,
        )
        self.simvp_recipe_effective = get_effective_simvp_recipe(
            self.arch,
            self.simvp_model_type,
            self.simvp_recipe,
        )

        if use_local_branch and self.arch != "hybrid_unet_facts":
            raise ValueError("The local F-region branch is only supported by 'hybrid_unet_facts'.")
        if self.arch == "predrnnpp" and self.predrnnpp_recipe not in PREDRNNPP_RECIPE_CHOICES:
            raise ValueError(
                f"Unsupported PredRNN++ recipe '{predrnnpp_recipe}'. Available choices: {PREDRNNPP_RECIPE_CHOICES}."
            )

        if self.arch == "simvp":
            self.backbone = SimVP(
                shape_in=(in_T, C, H, W),
                hid_S=hid_S,
                hid_T=hid_T,
                N_S=N_S,
                N_T=N_T,
                model_type=self.simvp_model_type,
                spatio_kernel_enc=simvp_spatio_kernel_enc,
                spatio_kernel_dec=simvp_spatio_kernel_dec,
                mlp_ratio=simvp_mlp_ratio,
                drop=simvp_drop,
                drop_path=simvp_drop_path,
            )
        elif self.arch == "tau":
            self.backbone = TAU_Model(
                shape_in=(in_T, C, H, W),
                hid_S=hid_S,
                hid_T=hid_T,
                N_S=N_S,
                N_T=N_T,
                spatio_kernel_enc=tau_spatio_kernel_enc,
                spatio_kernel_dec=tau_spatio_kernel_dec,
                mlp_ratio=tau_mlp_ratio,
                drop=tau_drop,
                drop_path=tau_drop_path,
            )
        elif self.arch == "earthfarseer":
            if isinstance(earthfarseer_incep_ker, str):
                earthfarseer_incep_ker = tuple(
                    int(part.strip())
                    for part in earthfarseer_incep_ker.split(",")
                    if part.strip()
                )
            else:
                earthfarseer_incep_ker = tuple(int(part) for part in earthfarseer_incep_ker)
            if not earthfarseer_incep_ker:
                raise ValueError("earthfarseer_incep_ker must contain at least one kernel size.")

            self.backbone = EarthFarseer_Model(
                shape_in=(in_T, C, H, W),
                hid_S=hid_S,
                hid_T=hid_T,
                N_S=N_S,
                N_T=N_T,
                incep_ker=earthfarseer_incep_ker,
                groups=earthfarseer_groups,
                num_interactions=earthfarseer_num_interactions,
                patch_size=earthfarseer_patch_size,
                embed_dim=earthfarseer_embed_dim,
                depth=earthfarseer_depth,
                spatial_depth=earthfarseer_spatial_depth,
                temporal_depth=earthfarseer_temporal_depth,
                mlp_ratio=earthfarseer_mlp_ratio,
                drop=earthfarseer_drop,
                drop_path=earthfarseer_drop_path,
                out_T=out_T,
            )
        elif self.arch == "convlstm":
            if convlstm_stride != 1:
                raise ValueError(
                    "ConvLSTM in simvp_spreadf only supports convlstm_stride=1. "
                    "stride>1 is not wired through the hidden-state spatial shapes, LayerNorm, "
                    "or output reconstruction path."
                )
            self.backbone = ConvLSTM_Model(
                shape_in=(in_T, C, H, W),
                out_T=out_T,
                num_hidden=convlstm_hidden,
                filter_size=convlstm_filter_size,
                patch_size=convlstm_patch_size,
                stride=convlstm_stride,
                layer_norm=convlstm_layer_norm,
            )
        elif self.arch == "mim":
            if mim_stride != 1:
                raise ValueError(
                    "MIM in simvp_spreadf only supports mim_stride=1. "
                    "stride>1 is not wired through the hidden/memory spatial shapes, LayerNorm, "
                    "or output reconstruction path."
                )
            self.backbone = MIM_Model(
                shape_in=(in_T, C, H, W),
                out_T=out_T,
                num_hidden=mim_hidden,
                filter_size=mim_filter_size,
                patch_size=mim_patch_size,
                stride=mim_stride,
                layer_norm=mim_layer_norm,
                reverse_scheduled_sampling=self.reverse_scheduled_sampling,
            )
        elif self.arch == "mau":
            if self.reverse_scheduled_sampling:
                raise ValueError(
                    "MAU in simvp_spreadf does not support reverse_scheduled_sampling. "
                    "Use standard scheduled sampling only."
                )
            if mau_stride != 1:
                raise ValueError(
                    "MAU in simvp_spreadf only supports mau_stride=1. "
                    "stride>1 is not wired through the recurrent-state or decoder shapes."
                )
            self.backbone = MAU_Model(
                shape_in=(in_T, C, H, W),
                out_T=out_T,
                num_hidden=mau_hidden,
                filter_size=mau_filter_size,
                patch_size=mau_patch_size,
                stride=mau_stride,
                sr_size=mau_sr_size,
                tau=mau_tau,
                cell_mode=mau_cell_mode,
                model_mode=mau_model_mode,
                layer_norm=mau_layer_norm,
                loss_mode=mau_loss_mode,
                conv_bias=mau_conv_bias,
            )
        elif self.arch == "predrnnpp":
            if predrnnpp_stride != 1:
                raise ValueError(
                    "PredRNN++ in simvp_spreadf only supports predrnnpp_stride=1. "
                    "stride>1 is not wired through the hidden/memory spatial shapes, LayerNorm, "
                    "or output reconstruction path."
                )
            self.backbone = PredRNNpp_Model(
                shape_in=(in_T, C, H, W),
                out_T=out_T,
                num_hidden=predrnnpp_hidden,
                filter_size=predrnnpp_filter_size,
                patch_size=predrnnpp_patch_size,
                stride=predrnnpp_stride,
                layer_norm=predrnnpp_layer_norm,
                reverse_scheduled_sampling=self.reverse_scheduled_sampling,
            )
        elif self.arch == "predrnnv2":
            if predrnnv2_stride != 1:
                raise ValueError(
                    "PredRNNv2 in simvp_spreadf only supports predrnnv2_stride=1. "
                    "stride>1 is not wired through the hidden/memory spatial shapes, LayerNorm, "
                    "or output reconstruction path."
                )
            self.backbone = PredRNNv2_Model(
                shape_in=(in_T, C, H, W),
                out_T=out_T,
                num_hidden=predrnnv2_hidden,
                filter_size=predrnnv2_filter_size,
                patch_size=predrnnv2_patch_size,
                stride=predrnnv2_stride,
                layer_norm=predrnnv2_layer_norm,
                reverse_scheduled_sampling=self.reverse_scheduled_sampling,
                decouple_beta=predrnnv2_decouple_beta,
            )
        elif self.arch == "predformer_facts":
            # predformer_depth remains the external config name for compatibility. In this
            # FacTS port it controls the shared stack depth used by both transformer branches.
            self.backbone = PredFormerFacTS_Model(
                shape_in=(in_T, C, H, W),
                patch_size=predformer_patch_size,
                dim=predformer_dim,
                heads=predformer_heads,
                dim_head=predformer_dim_head,
                dropout=predformer_dropout,
                attn_dropout=predformer_attn_dropout,
                drop_path=predformer_drop_path,
                scale_dim=predformer_scale_dim,
                depth=predformer_depth,
            )
        elif self.arch == "predformer_quadruplet_tsst":
            # Keep the public predformer_depth knob for compatibility. In the local TSST port
            # it maps to the number of stacked Quadruplet-TSST layers (official PredFormer's
            # Ndepth), while predformer_transformer_depth controls the depth inside each
            # GatedTransformer branch. Total GTB blocks = 4 * predformer_depth * predformer_transformer_depth.
            self.backbone = PredFormerQuadrupletTSST_Model(
                shape_in=(in_T, C, H, W),
                patch_size=predformer_patch_size,
                dim=predformer_dim,
                heads=predformer_heads,
                dim_head=predformer_dim_head,
                dropout=predformer_dropout,
                attn_dropout=predformer_attn_dropout,
                drop_path=predformer_drop_path,
                scale_dim=predformer_scale_dim,
                depth=predformer_depth,
                transformer_depth=predformer_transformer_depth,
            )
        elif self.arch == "hybrid_unet_facts":
            self.backbone = HybridUNetFacTS(
                in_T=in_T,
                out_T=out_T,
                in_channels=C,
                height=H,
                width=W,
                depth=hybrid_depth,
                heads=hybrid_heads,
                ffn_ratio=hybrid_ffn_ratio,
                attn_dropout=hybrid_attn_dropout,
                ffn_dropout=hybrid_ffn_dropout,
                drop_path=hybrid_drop_path,
                use_local_branch=use_local_branch,
                local_crop=local_crop,
            )
        else:
            raise ValueError(f"Unsupported arch '{arch}'. Available choices: {SUPPORTED_ARCHS}.")

    def forward(
        self,
        x,
        x_local=None,
        return_aux: bool = False,
        strict_local: bool = False,
        mask_true=None,
        return_loss: bool = False,
        loss_target=None,
    ):
        """
        x: [B, in_T, C, H, W]
        return: [B, out_T, C, H, W]
        """
        if return_loss and self.arch not in ("mau", "mim", "predrnnpp", "predrnnv2"):
            raise ValueError(
                "return_loss is only supported for arch='mau', 'mim', 'predrnnpp', or 'predrnnv2'."
            )

        if self.arch == "hybrid_unet_facts":
            y = self.backbone(x, x_local=x_local, return_aux=return_aux, strict_local=strict_local)
        elif self.arch == "predrnnpp":
            y = self.backbone(
                x,
                mask_true=mask_true,
                return_loss=return_loss,
                loss_target=loss_target,
                recipe=self.predrnnpp_recipe,
            )
        elif self.arch == "predrnnv2":
            y = self.backbone(
                x,
                mask_true=mask_true,
                return_loss=return_loss,
                loss_target=loss_target,
            )
        elif self.arch == "mim":
            y = self.backbone(
                x,
                mask_true=mask_true,
                return_loss=return_loss,
                loss_target=loss_target,
            )
        elif self.arch == "mau":
            y = self.backbone(
                x,
                mask_true=mask_true,
                return_loss=return_loss,
                loss_target=loss_target,
            )
        else:
            y = self.backbone(x)
        if self.arch in ("simvp", "tau"):
            y = y[:, :self.out_T]
            return y
        if self.arch == "earthfarseer":
            return y
        if self.arch in ("predformer_facts", "predformer_quadruplet_tsst"):
            if self.out_T <= x.shape[1]:
                return y[:, :self.out_T]

            pred_chunks = [y]
            generated = y.shape[1]
            cur_seq = y
            while generated < self.out_T:
                cur_seq = self.backbone(cur_seq)
                pred_chunks.append(cur_seq)
                generated += cur_seq.shape[1]
            return torch.cat(pred_chunks, dim=1)[:, :self.out_T]
        if self.arch in ("mau", "mim", "predrnnpp", "predrnnv2"):
            # OpenSTL-style recurrent training can emit the whole rollout sequence so the
            # internal loss can compare against every predicted step. The public wrapper keeps
            # the rest of simvp_spreadf stable by exposing only the requested forecast tail.
            if return_loss:
                y, loss = y
                return y[:, -self.out_T :], loss
            return y[:, -self.out_T :]
        return y
