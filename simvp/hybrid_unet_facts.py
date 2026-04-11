import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def sinusoidal_embedding(length: int, dim: int) -> torch.Tensor:
    if length <= 0 or dim <= 0:
        raise ValueError(f"Sinusoidal embedding expects positive length and dim, got {length}, {dim}.")

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    embedding = torch.zeros(length, dim, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term[: embedding[:, 1::2].shape[1]])
    return embedding.unsqueeze(0)


def build_temporal_pos_embed(num_frames: int, dim: int) -> torch.Tensor:
    return sinusoidal_embedding(num_frames, dim).unsqueeze(-1).unsqueeze(-1)


def build_spatial_pos_embed(height: int, width: int, dim: int) -> torch.Tensor:
    row_embed = sinusoidal_embedding(height, dim).permute(0, 2, 1).unsqueeze(-1)
    col_embed = sinusoidal_embedding(width, dim).permute(0, 2, 1).unsqueeze(-2)

    row_embed = row_embed.expand(-1, -1, -1, width)
    col_embed = col_embed.expand(-1, -1, height, -1)

    # Fixed separable 2D sinusoidal PE on the bottleneck grid.
    return (0.5 * (row_embed + col_embed)).unsqueeze(1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        apply_activation: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = _make_group_norm(out_channels)
        self.act = nn.SiLU(inplace=True) if apply_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            apply_activation=False,
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.out_norm = _make_group_norm(out_channels)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_norm(x + residual)
        x = self.out_act(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.block = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.block(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.pre = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1)
        self.block = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.pre(x)

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gate(x)) * self.value(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class GatedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(dim * ffn_ratio)

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, heads=heads, attn_dropout=attn_dropout)
        self.attn_drop = DropPath(drop_path)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFeedForward(dim=dim, hidden_dim=hidden_dim, dropout=ffn_dropout)
        self.ffn_drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.attn_norm(x)))
        x = x + self.ffn_drop(self.ffn(self.ffn_norm(x)))
        return x


class GatedTransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GatedTransformerBlock(
                    dim=dim,
                    heads=heads,
                    ffn_ratio=ffn_ratio,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class StrictFacTSTranslator(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        heads: int = 8,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.temporal_transformer = GatedTransformerStack(
            dim=dim,
            depth=depth,
            heads=heads,
            ffn_ratio=ffn_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            drop_path=drop_path,
        )
        self.spatial_transformer = GatedTransformerStack(
            dim=dim,
            depth=depth,
            heads=heads,
            ffn_ratio=ffn_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape

        # PredFormer-style Fac-T-S:
        # [B, T, C, H, W] -> [B * H * W, T, C]
        temporal_tokens = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, num_frames, channels)
        temporal_tokens = self.temporal_transformer(temporal_tokens)
        x = temporal_tokens.reshape(batch_size, height, width, num_frames, channels).permute(0, 3, 4, 1, 2)

        # Then [B, T, C, H, W] -> [B * T, H * W, C]
        spatial_tokens = x.permute(0, 1, 3, 4, 2).reshape(batch_size * num_frames, height * width, channels)
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        x = spatial_tokens.reshape(batch_size, num_frames, height, width, channels).permute(0, 1, 4, 2, 3)
        return x


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, stage_dims: Sequence[int]):
        super().__init__()
        if len(stage_dims) != 5:
            raise ValueError(f"Expected 5 encoder stages, but got {len(stage_dims)}.")

        self.stem = DownsampleBlock(in_channels, stage_dims[0])      # 448 -> 224
        self.down1 = DownsampleBlock(stage_dims[0], stage_dims[1])   # 224 -> 112
        self.down2 = DownsampleBlock(stage_dims[1], stage_dims[2])   # 112 -> 56
        self.down3 = DownsampleBlock(stage_dims[2], stage_dims[3])   # 56 -> 28
        self.down4 = DownsampleBlock(stage_dims[3], stage_dims[4])   # 28 -> 14

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_224 = self.stem(x)
        skip_112 = self.down1(skip_224)
        skip_56 = self.down2(skip_112)
        skip_28 = self.down3(skip_56)
        bottleneck = self.down4(skip_28)
        skips = [skip_224, skip_112, skip_56, skip_28]
        return bottleneck, skips


class FrameDecoder(nn.Module):
    def __init__(self, stage_dims: Sequence[int], out_channels: int):
        super().__init__()
        self.up14_to_28 = UpsampleBlock(stage_dims[4], stage_dims[3], stage_dims[3])
        self.up28_to_56 = UpsampleBlock(stage_dims[3], stage_dims[2], stage_dims[2])
        self.up56_to_112 = UpsampleBlock(stage_dims[2], stage_dims[1], stage_dims[1])
        self.up112_to_224 = UpsampleBlock(stage_dims[1], stage_dims[0], stage_dims[0])
        self.up224_to_448 = UpsampleBlock(stage_dims[0], 0, stage_dims[0])
        self.readout = nn.Conv2d(stage_dims[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        skip_224, skip_112, skip_56, skip_28 = skips
        x = self.up14_to_28(x, skip_28)
        x = self.up28_to_56(x, skip_56)
        x = self.up56_to_112(x, skip_112)
        x = self.up112_to_224(x, skip_224)
        x = self.up224_to_448(x)
        x = self.readout(x)
        return x


class HybridUNetFacTS(nn.Module):
    def __init__(
        self,
        in_T: int = 8,
        out_T: int = 8,
        in_channels: int = 3,
        height: int = 448,
        width: int = 448,
        stage_dims: Sequence[int] = (32, 64, 128, 256, 256),
        depth: int = 2,
        heads: int = 8,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError("HybridUNetFacTS expects image height and width to be divisible by 32.")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, but got {depth}.")
        if in_T != out_T:
            raise ValueError(
                "HybridUNetFacTS only supports equal-length input/output because its translator "
                f"is a recurrent-free hidden-state transform. Received in_T={in_T}, out_T={out_T}."
            )

        self.in_T = in_T
        self.out_T = out_T
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.stage_dims = tuple(stage_dims)
        self.bottleneck_height = height // 32
        self.bottleneck_width = width // 32

        self.encoder = FrameEncoder(in_channels=in_channels, stage_dims=self.stage_dims)
        self.translator = StrictFacTSTranslator(
            dim=self.stage_dims[-1],
            depth=depth,
            heads=heads,
            ffn_ratio=ffn_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            drop_path=drop_path,
        )
        self.decoder = FrameDecoder(stage_dims=self.stage_dims, out_channels=in_channels)

        self.register_buffer(
            "temporal_pos_embed",
            build_temporal_pos_embed(self.in_T, self.stage_dims[-1]),
            persistent=False,
        )
        self.register_buffer(
            "spatial_pos_embed",
            build_spatial_pos_embed(self.bottleneck_height, self.bottleneck_width, self.stage_dims[-1]),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_T != self.out_T:
            raise RuntimeError(
                "HybridUNetFacTS forward requires in_T == out_T. "
                f"Configured values are in_T={self.in_T}, out_T={self.out_T}."
            )

        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.in_T:
            raise ValueError(f"Expected {self.in_T} input frames, but got {num_frames}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, but got {channels}.")
        if height != self.height or width != self.width:
            raise ValueError(
                f"Expected input size {(self.height, self.width)}, but got {(height, width)}."
            )

        frames = x.reshape(batch_size * num_frames, channels, height, width)
        bottleneck, skips = self.encoder(frames)

        bottleneck = bottleneck.reshape(
            batch_size,
            num_frames,
            self.stage_dims[-1],
            self.bottleneck_height,
            self.bottleneck_width,
        )
        skips = [
            skip.reshape(batch_size, num_frames, skip.size(1), skip.size(2), skip.size(3))
            for skip in skips
        ]

        bottleneck = bottleneck + self.temporal_pos_embed.to(dtype=bottleneck.dtype)
        bottleneck = bottleneck + self.spatial_pos_embed.to(dtype=bottleneck.dtype)
        bottleneck = self.translator(bottleneck)

        bottleneck = bottleneck.reshape(
            batch_size * num_frames,
            self.stage_dims[-1],
            self.bottleneck_height,
            self.bottleneck_width,
        )
        decoded_skips = [
            skip.reshape(batch_size * num_frames, skip.size(2), skip.size(3), skip.size(4))
            for skip in skips
        ]

        y = self.decoder(bottleneck, decoded_skips)
        y = y.reshape(batch_size, num_frames, self.in_channels, height, width)
        return y
