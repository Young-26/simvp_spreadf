import torch.nn as nn

from .hybrid_unet_facts import HybridUNetFacTS
from .model import SimVP


SUPPORTED_ARCHS = ("simvp", "hybrid_unet_facts")


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
        arch: str = "simvp",
        hybrid_depth: int = 2,
        hybrid_heads: int = 8,
        hybrid_ffn_ratio: float = 4.0,
        hybrid_attn_dropout: float = 0.1,
        hybrid_ffn_dropout: float = 0.1,
        hybrid_drop_path: float = 0.1,
        use_local_branch: bool = False,
        local_crop: tuple[int, int] = (186, 410),
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

    def forward(self, x, x_local=None, return_aux: bool = False):
        """
        x: [B, in_T, C, H, W]
        return: [B, out_T, C, H, W]
        """
        if self.arch == "hybrid_unet_facts":
            y = self.backbone(x, x_local=x_local, return_aux=return_aux)
        else:
            y = self.backbone(x)
        if self.arch == "simvp":
            y = y[:, :self.out_T]
        return y
