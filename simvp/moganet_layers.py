import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAggregationFFN(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        mlp_hidden_dims: int,
        kernel_size: int = 3,
        act_layer=nn.GELU,
        ffn_drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Conv2d(embed_dims, mlp_hidden_dims, kernel_size=1)
        self.dwconv = nn.Conv2d(
            mlp_hidden_dims,
            mlp_hidden_dims,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
            groups=mlp_hidden_dims,
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2d(mlp_hidden_dims, embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(mlp_hidden_dims, 1, kernel_size=1)
        self.sigma = nn.Parameter(1e-5 * torch.ones((1, mlp_hidden_dims, 1, 1)), requires_grad=True)
        self.decompose_act = act_layer()

    def feat_decompose(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sigma * (x - self.decompose_act(self.decompose(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims: int, dw_dilation=(1, 2, 3), channel_split=(1, 3, 4)):
        super().__init__()
        if len(dw_dilation) != 3 or len(channel_split) != 3:
            raise ValueError("MogaNet expects exactly three dilation values and three channel-split ratios.")
        if embed_dims % sum(channel_split) != 0:
            raise ValueError(
                f"MogaNet embed_dims ({embed_dims}) must be divisible by the channel_split sum ({sum(channel_split)})."
            )

        split_ratio = [part / sum(channel_split) for part in channel_split]
        self.embed_dims_1 = int(split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims

        self.dw_conv0 = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=embed_dims,
            stride=1,
            dilation=dw_dilation[0],
        )
        self.dw_conv1 = nn.Conv2d(
            self.embed_dims_1,
            self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],
        )
        self.dw_conv2 = nn.Conv2d(
            self.embed_dims_2,
            self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],
        )
        self.pw_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.dw_conv0(x)
        x_1 = self.dw_conv1(x_0[:, self.embed_dims_0 : self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.dw_conv2(x_0[:, self.embed_dims - self.embed_dims_2 :, ...])
        x = torch.cat([x_0[:, : self.embed_dims_0, ...], x_1, x_2], dim=1)
        return self.pw_conv(x)


class MultiOrderGatedAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        attn_dw_dilation=(1, 2, 3),
        attn_channel_split=(1, 3, 4),
        attn_shortcut: bool = True,
    ):
        super().__init__()
        self.attn_shortcut = attn_shortcut
        self.proj_1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        self.sigma = nn.Parameter(1e-5 * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def feat_decompose(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_1(x)
        pooled = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma * (x - pooled)
        return self.act_value(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone() if self.attn_shortcut else None
        x = self.feat_decompose(x)
        gated = self.act_gate(self.gate(x)) * self.act_gate(self.value(x))
        x = self.proj_2(gated)
        if shortcut is not None:
            x = x + shortcut
        return x
