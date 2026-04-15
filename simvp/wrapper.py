import torch.nn as nn
from typing import Tuple

from .convlstm_model import ConvLSTM_Model
from .hybrid_unet_facts import HybridUNetFacTS
from .model import SimVP
from .predrnnpp_model import PredRNNpp_Model


SUPPORTED_ARCHS = ("simvp", "hybrid_unet_facts", "convlstm", "predrnnpp")


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
        convlstm_hidden: str = "128,128,128,128",
        convlstm_filter_size: int = 5,
        convlstm_patch_size: int = 4,
        convlstm_stride: int = 1,
        convlstm_layer_norm: bool = False,
        predrnnpp_hidden: str = "128,128,128,128",
        predrnnpp_filter_size: int = 5,
        predrnnpp_patch_size: int = 4,
        predrnnpp_stride: int = 1,
        predrnnpp_layer_norm: bool = False,
        arch: str = "simvp",
        hybrid_depth: int = 2,
        hybrid_heads: int = 8,
        hybrid_ffn_ratio: float = 4.0,
        hybrid_attn_dropout: float = 0.1,
        hybrid_ffn_dropout: float = 0.1,
        hybrid_drop_path: float = 0.1,
        use_local_branch: bool = False,
        local_crop: Tuple[int, int] = (186, 410),
    ):
        super().__init__()
        self.arch = arch.lower()
        self.out_T = out_T

        if use_local_branch and self.arch != "hybrid_unet_facts":
            raise ValueError("The local F-region branch is only supported by 'hybrid_unet_facts'.")

        if self.arch == "simvp":
            self.backbone = SimVP(
                shape_in=(in_T, C, H, W),
                hid_S=hid_S,
                hid_T=hid_T,
                N_S=N_S,
                N_T=N_T,
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

    def forward(self, x, x_local=None, return_aux: bool = False, strict_local: bool = False):
        """
        x: [B, in_T, C, H, W]
        return: [B, out_T, C, H, W]
        """
        if self.arch == "hybrid_unet_facts":
            y = self.backbone(x, x_local=x_local, return_aux=return_aux, strict_local=strict_local)
        else:
            y = self.backbone(x)
        if self.arch == "simvp":
            y = y[:, :self.out_T]
        return y
