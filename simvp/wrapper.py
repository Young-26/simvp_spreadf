import torch.nn as nn

from .hybrid_unet_facts import HybridUNetFacTS
from .model import SimVP


SUPPORTED_ARCHS = ("simvp", "hybrid_unet_facts")


def validate_hybrid_sequence_lengths(arch: str, in_T: int, out_T: int) -> None:
    arch = arch.lower()
    if arch == "hybrid_unet_facts" and in_T != out_T:
        raise ValueError(
            "HybridUNetFacTS only supports equal-length input/output because its translator "
            "is an equal-length hidden-state transform and does not include an explicit "
            f"temporal projection head. Received in_T={in_T}, out_T={out_T}."
        )


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
    ):
        super().__init__()
        self.arch = arch.lower()
        self.out_T = out_T
        validate_hybrid_sequence_lengths(self.arch, in_T, out_T)

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
            )
        else:
            raise ValueError(f"Unsupported arch '{arch}'. Available choices: {SUPPORTED_ARCHS}.")

    def forward(self, x):
        """
        x: [B, in_T, C, H, W]
        return: [B, out_T, C, H, W]
        """
        y = self.backbone(x)
        if self.arch == "simvp":
            y = y[:, :self.out_T]
        return y
