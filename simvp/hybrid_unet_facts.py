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


def _resolve_num_heads(dim: int, preferred_heads: int) -> int:
    heads = min(preferred_heads, dim)
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


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


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(attn_dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, dim = query.shape
        memory_len = memory.shape[1]

        q = self.q_proj(query).reshape(batch_size, query_len, self.heads, self.head_dim)
        k = self.k_proj(memory).reshape(batch_size, memory_len, self.heads, self.head_dim)
        v = self.v_proj(memory).reshape(batch_size, memory_len, self.heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(batch_size, query_len, dim)
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

        # Strict PredFormer-style Fac-T-S:
        # [B, T, C, H, W] -> [B * H * W, T, C]
        temporal_tokens = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, num_frames, channels)
        temporal_tokens = self.temporal_transformer(temporal_tokens)
        x = temporal_tokens.reshape(batch_size, height, width, num_frames, channels).permute(0, 3, 4, 1, 2)

        # Then [B, T, C, H, W] -> [B * T, H * W, C]
        spatial_tokens = x.permute(0, 1, 3, 4, 2).reshape(batch_size * num_frames, height * width, channels)
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        x = spatial_tokens.reshape(batch_size, num_frames, height, width, channels).permute(0, 1, 4, 2, 3)
        return x


class FutureCrossAttentionHead(nn.Module):
    def __init__(
        self,
        in_T: int,
        out_T: int,
        dim: int,
        heads: int,
        attn_dropout: float = 0.1,
        ffn_ratio: float = 2.0,
        ffn_dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if in_T <= 0 or out_T <= 0:
            raise ValueError(f"FutureCrossAttentionHead expects positive in_T/out_T, got {in_T}, {out_T}.")

        self.in_T = in_T
        self.out_T = out_T
        self.dim = dim
        self.query_embed = nn.Parameter(torch.randn(1, out_T, dim) * (dim ** -0.5))
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim=dim, heads=heads, attn_dropout=attn_dropout)
        self.cross_drop = DropPath(drop_path)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFeedForward(dim=dim, hidden_dim=int(dim * ffn_ratio), dropout=ffn_dropout)
        self.ffn_drop = DropPath(drop_path)
        self.register_buffer(
            "future_temporal_pos_embed",
            sinusoidal_embedding(out_T, dim),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.in_T:
            raise ValueError(f"Expected {self.in_T} temporal tokens, but got {num_frames}.")
        if channels != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, but got {channels}.")

        # Each spatial location keeps its own temporal memory:
        # [B, T, C, H, W] -> [B * H * W, T, C]
        memory = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, num_frames, channels)
        queries = self.query_embed + self.future_temporal_pos_embed.to(dtype=memory.dtype)
        queries = queries.expand(memory.size(0), -1, -1)

        future = queries + self.cross_drop(self.cross_attn(self.query_norm(queries), self.memory_norm(memory)))
        future = future + self.ffn_drop(self.ffn(self.ffn_norm(future)))

        # [B * H * W, out_T, C] -> [B, out_T, C, H, W]
        future = future.reshape(batch_size, height, width, self.out_T, channels).permute(0, 3, 4, 1, 2)
        return future


class TemporalConvForecastHead(nn.Module):
    def __init__(
        self,
        in_T: int,
        out_T: int,
        dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_T <= 0 or out_T <= 0:
            raise ValueError(f"TemporalConvForecastHead expects positive in_T/out_T, got {in_T}, {out_T}.")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}.")

        self.in_T = in_T
        self.out_T = out_T
        self.dim = dim

        self.input_norm = nn.LayerNorm(dim)
        self.depthwise_temporal = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.pointwise_temporal = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.temporal_act = nn.SiLU()

        self.context_norm = nn.LayerNorm(dim * 2)
        self.context_proj = nn.Linear(dim * 2, dim)
        self.future_bias = nn.Parameter(torch.zeros(1, out_T, dim))
        self.future_norm = nn.LayerNorm(dim)
        self.future_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.register_buffer(
            "past_temporal_pos_embed",
            sinusoidal_embedding(in_T, dim),
            persistent=False,
        )
        self.register_buffer(
            "future_temporal_pos_embed",
            sinusoidal_embedding(out_T, dim),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.in_T:
            raise ValueError(f"Expected {self.in_T} temporal tokens, but got {num_frames}.")
        if channels != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, but got {channels}.")

        # Lightweight skip forecasting:
        # [B, T, C, H, W] -> [B * H * W, T, C]
        tokens = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, num_frames, channels)
        tokens = tokens + self.past_temporal_pos_embed.to(dtype=tokens.dtype)
        tokens = self.input_norm(tokens)

        history = tokens.transpose(1, 2)  # [B * H * W, C, T]
        history = self.depthwise_temporal(history)
        history = self.temporal_act(history)
        history = self.pointwise_temporal(history)
        history = self.temporal_act(history)

        mean_context = history.mean(dim=-1)
        last_context = history[..., -1]
        context = torch.cat([mean_context, last_context], dim=-1)
        context = self.context_proj(self.context_norm(context))

        future = context.unsqueeze(1) + self.future_bias + self.future_temporal_pos_embed.to(dtype=context.dtype)
        future = future + self.future_mlp(self.future_norm(future))

        # [B * H * W, out_T, C] -> [B, out_T, C, H, W]
        future = future.reshape(batch_size, height, width, self.out_T, channels).permute(0, 3, 4, 1, 2)
        return future


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


class TAUChannelMixing(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            _make_group_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention(nn.Module):
    """Minimal TAU-style temporal attention for the local bottleneck translator."""

    def __init__(self, dim: int, kernel_size: int = 21, attn_shortcut: bool = True):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.temporal_gating_unit = TemporalAttentionModule(dim, kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attn_shortcut = attn_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x if self.attn_shortcut else None
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.temporal_gating_unit(x)
        x = self.proj_2(x)
        if shortcut is not None:
            x = x + shortcut
        return x


class TemporalAttentionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 3, reduction: int = 16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2

        self.conv0 = nn.Conv2d(dim, dim, kernel_size=d_k, padding=d_p, groups=dim, bias=False)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            kernel_size=dd_k,
            stride=1,
            padding=dd_p,
            groups=dim,
            dilation=dilation,
            bias=False,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        reduction_factor = max(dim // reduction, 4)
        reduced_dim = max(dim // reduction_factor, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        batch_size, channels, _, _ = x.shape
        se = self.avg_pool(x).view(batch_size, channels)
        se = self.fc(se).view(batch_size, channels, 1, 1)
        return se * attn * residual


class TAUBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        init_value: float = 1e-2,
    ):
        super().__init__()
        self.norm1 = _make_group_norm(dim)
        self.attn = TemporalAttention(dim, kernel_size=kernel_size)
        self.drop_path = DropPath(drop_path)

        self.norm2 = _make_group_norm(dim)
        self.mlp = TAUChannelMixing(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(
            self.layer_scale_1.view(1, -1, 1, 1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x))
        )
        return x


class TAUTranslator(nn.Module):
    def __init__(
        self,
        in_T: int,
        bottleneck_dim: int,
        depth: int = 2,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"TAUTranslator expects depth >= 1, but got {depth}.")

        dim = in_T * bottleneck_dim
        drop_rates = torch.linspace(0.0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TAUBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.in_T = in_T
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        if num_frames != self.in_T:
            raise ValueError(f"Expected {self.in_T} local frames, but got {num_frames}.")
        if channels != self.bottleneck_dim:
            raise ValueError(f"Expected local bottleneck dim {self.bottleneck_dim}, but got {channels}.")

        z = x.reshape(batch_size, num_frames * channels, height, width)
        for block in self.blocks:
            z = block(z)
        return z.reshape(batch_size, num_frames, channels, height, width)


class LocalFRegionBranch(nn.Module):
    """
    Lightweight local prior predictor for the fixed F-region crop.

    It consumes local past frames and predicts fusion-ready future priors at the
    decoder 1/8 and 1/4 scales instead of reconstructing full local images.
    """

    def __init__(
        self,
        in_T: int,
        out_T: int,
        in_channels: int,
        full_height: int,
        full_width: int,
        local_crop: Tuple[int, int] = (186, 410),
        branch_dims: Sequence[int] = (16, 32, 64),
        prior_1_4_channels: int = 64,
        prior_1_8_channels: int = 128,
        tau_depth: int = 2,
        tau_kernel_size: int = 21,
        tau_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if len(branch_dims) != 3:
            raise ValueError(f"LocalFRegionBranch expects 3 branch dims, but got {len(branch_dims)}.")

        top, bottom = local_crop
        if bottom <= top:
            raise ValueError(f"local_crop expects bottom > top, but got {local_crop}.")

        self.in_T = in_T
        self.out_T = out_T
        self.in_channels = in_channels
        self.full_height = full_height
        self.full_width = full_width
        self.local_crop = tuple(local_crop)
        self.local_height = bottom - top
        self.local_width = full_width
        self.branch_dims = tuple(branch_dims)

        self.stem = DownsampleBlock(in_channels, self.branch_dims[0])               # 224x448 -> 112x224
        self.down1 = DownsampleBlock(self.branch_dims[0], self.branch_dims[1])      # 112x224 -> 56x112
        self.down2 = DownsampleBlock(self.branch_dims[1], self.branch_dims[2])      # 56x112 -> 28x56
        self.temporal_module = TAUTranslator(
            in_T=in_T,
            bottleneck_dim=self.branch_dims[2],
            depth=tau_depth,
            kernel_size=tau_kernel_size,
            mlp_ratio=tau_mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.bottleneck_future_head = TemporalConvForecastHead(
            in_T=in_T,
            out_T=out_T,
            dim=self.branch_dims[2],
            kernel_size=3,
            dropout=dropout,
        )
        self.skip_1_4_future_head = TemporalConvForecastHead(
            in_T=in_T,
            out_T=out_T,
            dim=self.branch_dims[1],
            kernel_size=3,
            dropout=dropout,
        )

        self.up_1_8_to_1_4 = UpsampleBlock(self.branch_dims[2], self.branch_dims[1], self.branch_dims[1])
        self.prior_1_8_head = nn.Sequential(
            ResidualConvBlock(self.branch_dims[2], self.branch_dims[2]),
            nn.Conv2d(self.branch_dims[2], prior_1_8_channels, kernel_size=1),
        )
        self.prior_1_4_head = nn.Sequential(
            ResidualConvBlock(self.branch_dims[1], self.branch_dims[1]),
            nn.Conv2d(self.branch_dims[1], prior_1_4_channels, kernel_size=1),
        )

    def _project_to_global_scale(self, prior_local: torch.Tensor, scale: int) -> torch.Tensor:
        batch_size, num_frames, channels, _, _ = prior_local.shape
        target_height = self.local_height // scale
        target_width = self.full_width // scale
        global_height = self.full_height // scale
        global_width = self.full_width // scale

        if prior_local.shape[-2:] != (target_height, target_width):
            prior_local = F.interpolate(
                prior_local.reshape(batch_size * num_frames, channels, prior_local.size(-2), prior_local.size(-1)),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_frames, channels, target_height, target_width)

        # local_crop=(186, 410) is defined in the 448x448 global coordinate system.
        # After downsampling, the prior is placed back onto the matching global decoder grid.
        start_row = int(self.local_crop[0] / float(scale) + 0.5)
        start_row = min(max(start_row, 0), max(global_height - target_height, 0))

        canvas = prior_local.new_zeros(batch_size, num_frames, channels, global_height, global_width)
        canvas[:, :, :, start_row:start_row + target_height, :] = prior_local
        return canvas

    def forward(self, x_local: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_frames, channels, height, width = x_local.shape
        if num_frames != self.in_T:
            raise ValueError(f"Expected {self.in_T} local input frames, but got {num_frames}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} local channels, but got {channels}.")
        if height != self.local_height or width != self.local_width:
            raise ValueError(
                f"Expected local input size {(self.local_height, self.local_width)}, but got {(height, width)}."
            )

        frames = x_local.reshape(batch_size * num_frames, channels, height, width)
        skip_1_2 = self.stem(frames)
        skip_1_4 = self.down1(skip_1_2)
        bottleneck = self.down2(skip_1_4)

        skip_1_4 = skip_1_4.reshape(
            batch_size,
            num_frames,
            self.branch_dims[1],
            skip_1_4.size(-2),
            skip_1_4.size(-1),
        )
        bottleneck = bottleneck.reshape(
            batch_size,
            num_frames,
            self.branch_dims[2],
            bottleneck.size(-2),
            bottleneck.size(-1),
        )

        # TAU models the future local F-region dynamics on the compact 1/8 local grid.
        bottleneck = self.temporal_module(bottleneck)
        future_bottleneck = self.bottleneck_future_head(bottleneck)      # [B, out_T, C, 28, 56]
        future_skip_1_4 = self.skip_1_4_future_head(skip_1_4)            # [B, out_T, C, 56, 112]

        prior_1_8_local = self.prior_1_8_head(
            future_bottleneck.reshape(
                batch_size * self.out_T,
                self.branch_dims[2],
                future_bottleneck.size(-2),
                future_bottleneck.size(-1),
            )
        ).reshape(batch_size, self.out_T, -1, future_bottleneck.size(-2), future_bottleneck.size(-1))

        prior_1_4_local = self.up_1_8_to_1_4(
            future_bottleneck.reshape(
                batch_size * self.out_T,
                self.branch_dims[2],
                future_bottleneck.size(-2),
                future_bottleneck.size(-1),
            ),
            future_skip_1_4.reshape(
                batch_size * self.out_T,
                self.branch_dims[1],
                future_skip_1_4.size(-2),
                future_skip_1_4.size(-1),
            ),
        )
        prior_1_4_local = self.prior_1_4_head(prior_1_4_local).reshape(
            batch_size,
            self.out_T,
            -1,
            future_skip_1_4.size(-2),
            future_skip_1_4.size(-1),
        )

        return {
            "prior_1_8": self._project_to_global_scale(prior_1_8_local, scale=8),
            "prior_1_4": self._project_to_global_scale(prior_1_4_local, scale=4),
        }


class HybridUNetFacTS(nn.Module):
    def __init__(
        self,
        in_T: int = 8,
        out_T: int = 2,
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
        use_local_branch: bool = False,
        local_crop: Tuple[int, int] = (186, 410),
        local_branch_dims: Sequence[int] = (16, 32, 64),
        local_branch_depth: int = 2,
    ):
        super().__init__()
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError("HybridUNetFacTS expects image height and width to be divisible by 32.")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, but got {depth}.")
        if in_T <= 0 or out_T <= 0:
            raise ValueError(f"HybridUNetFacTS expects positive in_T/out_T, got {in_T}, {out_T}.")

        self.in_T = in_T
        self.out_T = out_T
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.stage_dims = tuple(stage_dims)
        self.bottleneck_height = height // 32
        self.bottleneck_width = width // 32
        self.use_local_branch = bool(use_local_branch)
        self.local_crop = tuple(local_crop)

        self.encoder = FrameEncoder(in_channels=in_channels, stage_dims=self.stage_dims)
        self.translator = StrictFacTSTranslator(
            dim=self.stage_dims[-1],
            depth=depth,
            heads=_resolve_num_heads(self.stage_dims[-1], heads),
            ffn_ratio=ffn_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            drop_path=drop_path,
        )
        self.bottleneck_future_head = FutureCrossAttentionHead(
            in_T=in_T,
            out_T=out_T,
            dim=self.stage_dims[-1],
            heads=_resolve_num_heads(self.stage_dims[-1], heads),
            attn_dropout=attn_dropout,
            ffn_ratio=max(2.0, ffn_ratio / 2.0),
            ffn_dropout=ffn_dropout,
            drop_path=drop_path,
        )
        self.skip_future_heads = nn.ModuleList(
            [
                TemporalConvForecastHead(
                    in_T=in_T,
                    out_T=out_T,
                    dim=skip_dim,
                    kernel_size=3,
                    dropout=ffn_dropout,
                )
                for skip_dim in self.stage_dims[:-1]
            ]
        )
        self.decoder = FrameDecoder(stage_dims=self.stage_dims, out_channels=in_channels)

        if self.use_local_branch:
            self.local_branch = LocalFRegionBranch(
                in_T=in_T,
                out_T=out_T,
                in_channels=in_channels,
                full_height=height,
                full_width=width,
                local_crop=self.local_crop,
                branch_dims=local_branch_dims,
                prior_1_4_channels=self.stage_dims[1],
                prior_1_8_channels=self.stage_dims[2],
                tau_depth=local_branch_depth,
                dropout=ffn_dropout,
                drop_path=drop_path,
            )
            self.local_prior_scale_1_4 = nn.Parameter(torch.tensor(1.0))
            self.local_prior_scale_1_8 = nn.Parameter(torch.tensor(1.0))
        else:
            self.local_branch = None

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

    def forward(
        self,
        x: torch.Tensor,
        x_local: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        # Translator keeps equal-length hidden states. Future forecasting starts here.
        future_bottleneck = self.bottleneck_future_head(bottleneck)
        future_skips = [
            head(skip)
            for head, skip in zip(self.skip_future_heads, skips)
        ]

        aux_outputs: dict[str, torch.Tensor] = {}
        if self.local_branch is not None and x_local is not None:
            local_priors = self.local_branch(x_local)
            # The local branch predicts future F-region priors and injects them into
            # the global decoder's 1/8 and 1/4 skip features.
            future_skips[2] = future_skips[2] + self.local_prior_scale_1_8 * local_priors["prior_1_8"]
            future_skips[1] = future_skips[1] + self.local_prior_scale_1_4 * local_priors["prior_1_4"]
            aux_outputs.update(local_priors)

        future_bottleneck = future_bottleneck.reshape(
            batch_size * self.out_T,
            self.stage_dims[-1],
            self.bottleneck_height,
            self.bottleneck_width,
        )
        decoded_skips = [
            skip.reshape(batch_size * self.out_T, skip.size(2), skip.size(3), skip.size(4))
            for skip in future_skips
        ]

        y = self.decoder(future_bottleneck, decoded_skips)
        y = y.reshape(batch_size, self.out_T, self.in_channels, height, width)
        if return_aux:
            return y, aux_outputs
        return y
