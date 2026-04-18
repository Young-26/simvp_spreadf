from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .model import Decoder as SimVPDecoder
from .model import Encoder as SimVPEncoder
from .model import Mid_Xnet
from .modules import Inception
from .tau_model import DropPath


def _to_2tuple(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, but received {value}.")
        return int(value[0]), int(value[1])
    scalar = int(value)
    return scalar, scalar


def _resolve_depth(explicit_depth: Optional[int], legacy_depth: int, *, name: str) -> int:
    resolved = legacy_depth if explicit_depth is None else explicit_depth
    resolved = int(resolved)
    if resolved < 1:
        raise ValueError(f"EarthFarseer {name} must be >= 1, but got {resolved}.")
    return resolved


def _resolve_group_norm_groups(num_channels: int, max_groups: int) -> int:
    num_channels = int(num_channels)
    max_groups = max(1, min(int(max_groups), num_channels))
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ResidualSkipConnection(nn.Module):
    def __init__(
        self,
        shape_in,
        hid_S: int = 16,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        incep_ker: Sequence[int] = (3, 5, 7, 11),
        groups: int = 8,
    ):
        super().__init__()
        T, C, _, _ = shape_in
        self.enc = SimVPEncoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, list(incep_ker), groups)
        self.dec = SimVPDecoder(hid_S, C, N_S)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B * T, C, H, W)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        hidden = self.hid(embed.reshape(B, T, C_, H_, W_))
        hidden = hidden.reshape(B * T, C_, H_, W_)
        y = self.dec(hidden, skip)
        return y.reshape(B, T, C, H, W)


class FourierMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim: int, h: int, w: int, bias: bool = True):
        super().__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = 2
        self.block_size = self.hidden_size // self.num_blocks
        if self.hidden_size % self.num_blocks != 0:
            raise ValueError(
                f"EarthFarseer Fourier hidden size must be divisible by {self.num_blocks}, got {self.hidden_size}."
            )

        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, self.num_blocks, self.block_size))
        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1) if bias else None
        self.softshrink = 0.0

    @staticmethod
    def multiply(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...bd,bdk->...bk", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        residual = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1) if self.bias is not None else 0.0

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        old_real = x.real
        old_imag = x.imag
        x_real = F.relu(
            self.multiply(old_real, self.w1[0]) - self.multiply(old_imag, self.w1[1]) + self.b1[0],
            inplace=True,
        )
        x_imag = F.relu(
            self.multiply(old_real, self.w1[1]) + self.multiply(old_imag, self.w1[0]) + self.b1[1],
            inplace=True,
        )
        # Keep real/imag updates within the same stage on the same snapshot to avoid order-dependent pollution.
        old_real = x_real
        old_imag = x_imag
        x_real = self.multiply(old_real, self.w2[0]) - self.multiply(old_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(old_real, self.w2[1]) + self.multiply(old_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        if self.softshrink > 0.0:
            x = F.softshrink(x, lambd=self.softshrink)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        return x + residual


class FourierNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        h: int = 14,
        w: int = 14,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FourierMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.filter(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Sequence[int]],
        patch_size: Union[int, Sequence[int]],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        if self.img_size[0] % self.patch_size[0] != 0 or self.img_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"EarthFarseer patch embedding requires image size divisible by patch size, got "
                f"img_size={self.img_size}, patch_size={self.patch_size}."
            )
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(
                f"EarthFarseer PatchEmbed expected spatial size {self.img_size}, but received {(H, W)}."
            )
        return self.projection(x).flatten(2).transpose(1, 2)


class GlobalFourierBlock(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Sequence[int]],
        patch_size: int,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        depth: int,
        mlp_ratio: float,
        drop: float,
        drop_path: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)
        self.h, self.w = self.patch_embed.grid_size

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)] if depth > 0 else []
        self.blocks = nn.ModuleList(
            [
                FourierNetBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    h=self.h,
                    w=self.w,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if patch_size == 16:
            self.recovery = nn.Sequential(
                OrderedDict(
                    [
                        ("transposeconv1", nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=2, stride=2)),
                        ("act1", nn.Tanh()),
                        ("transposeconv2", nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=2, stride=2)),
                        ("act2", nn.Tanh()),
                        ("transposeconv3", nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=4, stride=4)),
                    ]
                )
            )
        else:
            self.recovery = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels * patch_size * patch_size, kernel_size=1),
                nn.GELU(),
                nn.PixelShuffle(patch_size),
            )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).transpose(1, 2)
        return x.reshape(B * T, self.embed_dim, self.h, self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.recovery(x)
        return x.reshape(B, T, C, H, W)


class LocalCNNBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 3,
        norm_groups: int = 8,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"EarthFarseer local branch depth must be >= 1, but got {num_layers}.")

        hidden_channels = max(int(hidden_channels), in_channels, out_channels)
        stem_groups = _resolve_group_norm_groups(hidden_channels, norm_groups)

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(stem_groups, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(1, num_layers):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(stem_groups, hidden_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        self.body = nn.Sequential(*layers)
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        residual = self.shortcut(x)
        x = self.body(x)
        x = self.readout(x)
        x = x + residual
        return x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])


class TemporalProjectionHead(nn.Module):
    def __init__(self, in_steps: int, out_steps: int):
        super().__init__()
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        if self.in_steps < 1 or self.out_steps < 1:
            raise ValueError(
                f"EarthFarseer temporal projection requires positive step counts, got in={in_steps}, out={out_steps}."
            )
        self.proj = nn.Identity() if self.in_steps == self.out_steps else nn.Linear(self.in_steps, self.out_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_steps == self.out_steps:
            return x
        # Project along the time axis per spatial location/channel instead of relying on wrapper-side truncation.
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.proj(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class FoTF(nn.Module):
    def __init__(
        self,
        shape_in,
        num_interactions: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        spatial_depth: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_hidden_channels: int = 16,
        local_depth: int = 3,
    ):
        super().__init__()
        T, C, H, W = shape_in
        self.local_branch = LocalCNNBranch(
            in_channels=C,
            out_channels=C,
            hidden_channels=local_hidden_channels,
            num_layers=local_depth,
        )
        self.global_branch = GlobalFourierBlock(
            img_size=(H, W),
            patch_size=patch_size,
            in_channels=C,
            out_channels=C,
            embed_dim=embed_dim,
            depth=spatial_depth,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )
        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_raw.shape
        gf_features = self.global_branch(x_raw)
        lc_features = self.local_branch(x_raw)

        for _ in range(self.num_interactions):
            gf_up = self.up(gf_features.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
            lc_proj = self.conv1x1(lc_features.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
            combined = gf_up + lc_proj

            gf_features = self.global_branch(combined)
            lc_features = self.local_branch(combined)

            gf_down = self.down(gf_features.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
            lc_proj = self.conv1x1(lc_features.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
            combined = gf_down + lc_proj

            gf_features = self.global_branch(combined)
            lc_features = self.local_branch(combined)

        return gf_features + lc_features


class TemporalEvolutionBlock(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        N_T: int,
        h: int,
        w: int,
        temporal_depth: int,
        incep_ker: Sequence[int] = (3, 5, 7, 11),
        groups: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if N_T < 2:
            raise ValueError(f"EarthFarseer temporal block requires N_T >= 2, but got {N_T}.")
        self.N_T = N_T

        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=list(incep_ker), groups=groups)]
        for _ in range(1, N_T - 1):
            enc_layers.append(
                Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=list(incep_ker), groups=groups)
            )
        enc_layers.append(
            Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=list(incep_ker), groups=groups)
        )

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=list(incep_ker), groups=groups)]
        for _ in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=list(incep_ker), groups=groups)
            )
        dec_layers.append(
            Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=list(incep_ker), groups=groups)
        )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)
        self.enc = nn.Sequential(*enc_layers)
        self.blocks = nn.ModuleList(
            [
                FourierNetBlock(
                    dim=channel_hid,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    h=h,
                    w=w,
                )
                for _ in range(temporal_depth)
            ]
        )
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        residual = x
        x = x.reshape(B, T * C, H, W)

        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        B, D, H_lat, W_lat = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B, H_lat * W_lat, D)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z).permute(0, 2, 1).reshape(B, D, H_lat, W_lat)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y + residual


class EarthFarseer_Model(nn.Module):
    def __init__(
        self,
        shape_in,
        hid_S: int = 512,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        incep_ker: Sequence[int] = (3, 5, 7, 11),
        groups: int = 8,
        num_interactions: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        spatial_depth: Optional[int] = None,
        temporal_depth: Optional[int] = None,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        out_T: Optional[int] = None,
    ):
        super().__init__()
        T, C, H, W = shape_in
        resolved_out_T = T if out_T is None else int(out_T)
        # Keep the legacy single depth knob working while allowing separate spatial/temporal control.
        spatial_depth = _resolve_depth(spatial_depth, depth, name="spatial depth")
        temporal_depth = _resolve_depth(temporal_depth, depth, name="temporal depth")
        self.in_T = T
        self.out_T = resolved_out_T
        self.fotf_encoder = FoTF(
            shape_in=shape_in,
            num_interactions=num_interactions,
            patch_size=patch_size,
            embed_dim=embed_dim,
            spatial_depth=spatial_depth,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )
        self.skip_connection = ResidualSkipConnection(
            shape_in=shape_in,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            incep_ker=incep_ker,
            groups=groups,
        )
        self.latent_projection = SimVPEncoder(C, hid_S, N_S)
        self.dec = SimVPDecoder(hid_S, C, N_S)
        self.temporal_head = TemporalProjectionHead(T, resolved_out_T)

        with torch.no_grad():
            latent, _ = self.latent_projection(torch.zeros(1, C, H, W))
        h_lat, w_lat = latent.shape[-2:]

        self.temporal_block = TemporalEvolutionBlock(
            T * hid_S,
            hid_T,
            N_T,
            h_lat,
            w_lat,
            temporal_depth=temporal_depth,
            incep_ker=incep_ker,
            groups=groups,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        skip_feature = self.skip_connection(x)
        spatial_feature = self.fotf_encoder(x)
        spatial_feature = spatial_feature.reshape(B * T, C, H, W)

        spatial_embed, spatial_skip = self.latent_projection(spatial_feature)
        _, C_, H_, W_ = spatial_embed.shape
        spatial_embed = spatial_embed.reshape(B, T, C_, H_, W_)

        spatiotemporal_embed = self.temporal_block(spatial_embed)
        spatiotemporal_embed = spatiotemporal_embed.reshape(B * T, C_, H_, W_)

        predictions = self.dec(spatiotemporal_embed, spatial_skip)
        predictions = predictions.reshape(B, T, C, H, W)
        predictions = predictions + skip_feature
        return self.temporal_head(predictions)
