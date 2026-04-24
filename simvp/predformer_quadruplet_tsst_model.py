import torch
from torch import nn

from .predformer_facts_model import (
    GatedTransformer,
    PatchEmbed,
    _init_linear_and_layernorm,
    _recover_from_patch_tokens,
    _require_positive_int,
    _validate_shape_in,
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

        x = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * num_patches, steps, dim)
        x = self.ts_temporal_transformer(x)

        x = x.reshape(batch_size, num_patches, steps, dim).permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * steps, num_patches, dim)
        x = self.ts_space_transformer(x)

        x = x.reshape(batch_size, steps, num_patches, dim)
        x = x.reshape(batch_size * steps, num_patches, dim)
        x = self.st_space_transformer(x)

        x = x.reshape(batch_size, steps, num_patches, dim).permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_patches, steps, dim)
        x = self.st_temporal_transformer(x)

        return x.reshape(batch_size, num_patches, steps, dim).permute(0, 2, 1, 3).contiguous()


class PredFormerQuadrupletTSST_Model(nn.Module):
    """
    Local port of PredFormer_Quadruplet_TSST for simvp_spreadf.

    Official PredFormer naming uses `Ndepth` for the number of TSST layers and `depth`
    for the GatedTransformer depth inside each TS/ST branch. This port keeps the public
    `predformer_depth` interface compatible by mapping it to TSST layer count here.
    Total GTB blocks = 4 * depth * transformer_depth.
    """

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
        model_name = "PredFormer_Quadruplet_TSST"
        steps, channels, height, width = _validate_shape_in(shape_in, model_name)
        patch_size = _require_positive_int("patch_size", patch_size, model_name)
        dim = _require_positive_int("dim", dim, model_name)
        heads = _require_positive_int("heads", heads, model_name)
        dim_head = _require_positive_int("dim_head", dim_head, model_name)
        scale_dim = _require_positive_int("scale_dim", scale_dim, model_name)
        tsst_depth = _require_positive_int("depth", depth, model_name)
        transformer_depth = _require_positive_int("transformer_depth", transformer_depth, model_name)
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
        self.depth = tsst_depth
        self.transformer_depth = transformer_depth
        self.total_gtb_blocks = 4 * self.depth * self.transformer_depth
        self.patches_h = self.height // self.patch_size
        self.patches_w = self.width // self.patch_size
        self.num_patches = self.patches_h * self.patches_w

        self.to_patch_embedding = PatchEmbed(self.channels, self.patch_size, self.dim)
        pos_embedding = sinusoidal_embedding(self.steps * self.num_patches, self.dim)
        self.register_buffer(
            "pos_embedding",
            pos_embedding.view(1, self.steps, self.num_patches, self.dim),
        )

        mlp_dim = self.dim * scale_dim
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
                for _ in range(self.depth)
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
