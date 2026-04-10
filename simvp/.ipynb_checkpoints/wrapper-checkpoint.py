import torch
import torch.nn as nn

from .model import SimVP


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
    ):
        super().__init__()
        self.out_T = out_T
        self.backbone = SimVP(
            shape_in=(in_T, C, H, W),
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
        )

    def forward(self, x):
        """
        x: [B, 8, C, H, W]
        return: [B, 2, C, H, W]
        """
        y = self.backbone(x)      # 原始输出 [B, 8, C, H, W]
        y = y[:, :self.out_T]     # 只取前2帧
        return y