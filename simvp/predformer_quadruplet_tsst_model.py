import torch
from torch import nn

from .predformer_facts_model import (
    GatedTransformer,
    PatchEmbed,
    _init_linear_and_layernorm,
    sinusoidal_embedding,
)


class PredFormerQuadrupletTSSTLayer(nn.Module):
    """One TSST block: temporal -> spatial -> spatial -> temporal."""

    def __init__(
        self,
        dim: int,
        transformer_depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ts_temporal_transformer = GatedTransformer(
            dim,
            depth=transformer_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )
        self.ts_space_transformer = GatedTransformer(
            dim,
            depth=transformer_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )
        self.st_space_transformer = GatedTransformer(
            dim,
            depth=transformer_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )
        self.st_temporal_transformer = GatedTransformer(
            dim,
            depth=transformer_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, num_patches, dim = x.shape

        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * num_patches, steps, dim)
        x = self.ts_temporal_transformer(x)

        x = x.view(batch_size, num_patches, steps, dim).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * steps, num_patches, dim)
        x = self.ts_space_transformer(x)

        x = x.view(batch_size, steps, num_patches, dim)
        x = x.view(batch_size * steps, num_patches, dim)
        x = self.st_space_transformer(x)

        x = x.view(batch_size, steps, num_patches, dim).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * num_patches, steps, dim)
        x = self.st_temporal_transformer(x)

        return x.view(batch_size, num_patches, steps, dim).permute(0, 2, 1, 3).contiguous()


class PredFormerQuadrupletTSST_Model(nn.Module):
    """Local port of PredFormer_Quadruplet_TSST for simvp_spreadf."""

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
        depth: int = 6,
        transformer_depth: int = 1,
    ):
        super().__init__()
        steps, channels, height, width = shape_in
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "PredFormer_Quadruplet_TSST requires H/W divisible by patch_size, "
                f"got {(height, width)} and patch_size={patch_size}."
            )

        self.steps = int(steps)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.patches_h = self.height // self.patch_size
        self.patches_w = self.width // self.patch_size
        self.num_patches = self.patches_h * self.patches_w

        self.to_patch_embedding = PatchEmbed(self.channels, self.patch_size, self.dim)
        pos_embedding = sinusoidal_embedding(self.steps * self.num_patches, self.dim)
        self.register_buffer(
            "pos_embedding",
            pos_embedding.view(1, self.steps, self.num_patches, self.dim),
        )

        mlp_dim = self.dim * int(scale_dim)
        transformer_depth = int(transformer_depth)
        self.blocks = nn.ModuleList(
            [
                PredFormerQuadrupletTSSTLayer(
                    self.dim,
                    transformer_depth=transformer_depth,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.channels * self.patch_size * self.patch_size),
        )
        self.apply(_init_linear_and_layernorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, height, width = x.shape
        if steps != self.steps:
            raise ValueError(
                f"PredFormer_Quadruplet_TSST was built for T={self.steps}, but received T={steps}."
            )
        if channels != self.channels or height != self.height or width != self.width:
            raise ValueError(
                "PredFormer_Quadruplet_TSST received input with a different C/H/W than configured: "
                f"expected {(self.channels, self.height, self.width)}, got {(channels, height, width)}."
            )

        x = self.to_patch_embedding(x)
        pos_embedding = self.pos_embedding
        if pos_embedding.dtype != x.dtype:
            pos_embedding = pos_embedding.to(dtype=x.dtype)
        x = x + pos_embedding

        for block in self.blocks:
            x = block(x)

        x = self.head(x.reshape(-1, self.dim))
        x = x.view(
            batch_size,
            steps,
            self.patches_h,
            self.patches_w,
            self.channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(batch_size, steps, self.channels, self.height, self.width)
