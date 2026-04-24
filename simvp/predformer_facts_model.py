import math
from numbers import Integral

import torch
from torch import nn

from .tau_model import DropPath


def _init_linear_and_layernorm(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def _require_positive_int(name: str, value, model_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{model_name} expects {name} to be a positive integer, but got {value!r}.")
    return int(value)


def _validate_shape_in(shape_in, model_name: str):
    if not isinstance(shape_in, (tuple, list)) or len(shape_in) != 4:
        raise ValueError(f"{model_name} expects shape_in=(T, C, H, W), but got {shape_in!r}.")
    steps, channels, height, width = shape_in
    return (
        _require_positive_int("T", steps, model_name),
        _require_positive_int("C", channels, model_name),
        _require_positive_int("H", height, model_name),
        _require_positive_int("W", width, model_name),
    )


def _recover_from_patch_tokens(
    x: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    patches_h: int,
    patches_w: int,
    channels: int,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    x = x.reshape(
        batch_size,
        steps,
        patches_h,
        patches_w,
        channels,
        patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(batch_size, steps, channels, height, width)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.reshape(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2).contiguous()
            for tensor in qkv
        ]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
        return self.to_out(out)


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.fc1_g = nn.Linear(in_features, hidden_features)
        self.fc1_x = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1_g(x)) * self.fc1_x(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class GatedTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                        PreNorm(dim, SwiGLU(dim, hidden_features=mlp_dim, drop=dropout)),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


def sinusoidal_embedding(num_positions: int, dim: int) -> torch.Tensor:
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(1, num_positions, dim, dtype=torch.float32)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
    return pe


class PatchEmbed(nn.Module):
    def __init__(self, channels: int, patch_size: int, dim: int):
        super().__init__()
        self.channels = _require_positive_int("channels", channels, "PatchEmbed")
        self.patch_size = _require_positive_int("patch_size", patch_size, "PatchEmbed")
        self.proj = nn.Linear(
            self.channels * self.patch_size * self.patch_size,
            _require_positive_int("dim", dim, "PatchEmbed"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, height, width = x.shape
        patch_size = self.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"PatchEmbed requires H and W divisible by patch_size, but got "
                f"H={height}, W={width}, patch_size={patch_size}."
            )
        patches_h = height // patch_size
        patches_w = width // patch_size
        x = x.reshape(batch_size, steps, channels, patches_h, patch_size, patches_w, patch_size)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        x = x.reshape(batch_size, steps, patches_h * patches_w, channels * patch_size * patch_size)
        return self.proj(x)


class PredFormerFacTS_Model(nn.Module):
    """PredFormer FacTS port with a shared transformer depth for temporal and spatial stacks."""

    def __init__(
        self,
        shape_in,
        patch_size: int = 16,
        dim: int = 256,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        scale_dim: int = 4,
        depth: int = 4,
    ):
        super().__init__()
        model_name = "PredFormer_FacTS"
        steps, channels, height, width = _validate_shape_in(shape_in, model_name)
        patch_size = _require_positive_int("patch_size", patch_size, model_name)
        dim = _require_positive_int("dim", dim, model_name)
        heads = _require_positive_int("heads", heads, model_name)
        dim_head = _require_positive_int("dim_head", dim_head, model_name)
        scale_dim = _require_positive_int("scale_dim", scale_dim, model_name)
        shared_depth = _require_positive_int("depth", depth, model_name)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"{model_name} requires H and W divisible by patch_size, but got "
                f"H={height}, W={width}, patch_size={patch_size}."
            )

        self.steps = steps
        self.channels = channels
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.dim = dim
        self.patches_h = self.height // self.patch_size
        self.patches_w = self.width // self.patch_size
        self.num_patches = self.patches_h * self.patches_w

        self.to_patch_embedding = PatchEmbed(self.channels, self.patch_size, self.dim)
        pos_embedding = sinusoidal_embedding(self.steps * self.num_patches, self.dim)
        self.register_buffer(
            "pos_embedding",
            pos_embedding.view(1, self.steps, self.num_patches, self.dim),
        )

        # Keep the public predformer_depth setting compatible with existing CLI/config usage.
        # In this FacTS port it is the shared temporal/spatial transformer depth.
        mlp_dim = self.dim * scale_dim
        self.temporal_transformer = GatedTransformer(
            self.dim,
            depth=shared_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )
        self.spatial_transformer = GatedTransformer(
            self.dim,
            depth=shared_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.channels * self.patch_size * self.patch_size),
        )
        self.apply(_init_linear_and_layernorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, height, width = x.shape
        if steps != self.steps:
            raise ValueError(f"PredFormer_FacTS was built for T={self.steps}, but received T={steps}.")
        if channels != self.channels or height != self.height or width != self.width:
            raise ValueError(
                "PredFormer_FacTS received input with a different C/H/W than configured: "
                f"expected {(self.channels, self.height, self.width)}, got {(channels, height, width)}."
            )

        x = self.to_patch_embedding(x)
        pos_embedding = self.pos_embedding
        if pos_embedding.dtype != x.dtype:
            # The buffer already tracks device with the module; only align dtype under AMP to avoid
            # promoting activations back to fp32 during the addition.
            pos_embedding = pos_embedding.to(dtype=x.dtype)
        x = x + pos_embedding

        x_t = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * self.num_patches, steps, self.dim)
        x_t = self.temporal_transformer(x_t)

        x_ts = x_t.reshape(batch_size, self.num_patches, steps, self.dim).permute(0, 2, 1, 3).contiguous()
        x_ts = x_ts.reshape(batch_size * steps, self.num_patches, self.dim)
        x_ts = self.spatial_transformer(x_ts)

        x = self.head(x_ts.reshape(batch_size, steps, self.num_patches, self.dim).reshape(-1, self.dim))
        return _recover_from_patch_tokens(
            x,
            batch_size=batch_size,
            steps=steps,
            patches_h=self.patches_h,
            patches_w=self.patches_w,
            channels=self.channels,
            patch_size=self.patch_size,
            height=self.height,
            width=self.width,
        )
