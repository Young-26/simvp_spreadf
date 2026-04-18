import math

import torch
from torch import nn

from .modules import ConvSC, Inception
from .simvp_config import normalize_simvp_model_type
from .tau_model import Decoder as MetaDecoder
from .tau_model import DropPath, Encoder as MetaEncoder, MixMlp


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Mid_Xnet, self).__init__()
        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, N_T - 1):
            enc_layers.append(
                Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)
            )
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)
            )
        dec_layers.append(
            Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups)
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class AttentionModule(nn.Module):
    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            dd_k,
            stride=1,
            padding=dd_p,
            groups=dim,
            dilation=dilation,
        )
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        shortcut = x if self.attn_shortcut else None
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if shortcut is not None:
            x = x + shortcut
        return x


class GASubBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=21,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
        init_value=1e-2,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
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
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x


class MetaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_type="gsta",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if model_type != "gsta":
            raise ValueError(
                f"Unsupported SimVP MetaBlock model_type '{model_type}'. Only 'gsta' is implemented."
            )

        self.block = GASubBlock(
            in_channels,
            kernel_size=21,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
        )
        self.reduction = None
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        if self.reduction is not None:
            z = self.reduction(z)
        return z


class MidMetaNet(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_hid,
        N_T,
        model_type="gsta",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        if N_T < 2:
            raise ValueError(f"SimVP MetaNet requires N_T >= 2, but got {N_T}.")
        self.N_T = N_T
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N_T)]

        enc_layers = [
            MetaBlock(
                channel_in,
                channel_hid,
                model_type=model_type,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=dpr[0],
            )
        ]
        for i in range(1, N_T - 1):
            enc_layers.append(
                MetaBlock(
                    channel_hid,
                    channel_hid,
                    model_type=model_type,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[i],
                )
            )
        enc_layers.append(
            MetaBlock(
                channel_hid,
                channel_in,
                model_type=model_type,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
            )
        )
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        z = x.reshape(B, T * C, H, W)
        for i in range(self.N_T):
            z = self.enc[i](z)
        return z.reshape(B, T, C, H, W)


class SimVP(nn.Module):
    def __init__(
        self,
        shape_in,
        hid_S=16,
        hid_T=256,
        N_S=4,
        N_T=8,
        incep_ker=[3, 5, 7, 11],
        groups=8,
        model_type="incepu",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
    ):
        super(SimVP, self).__init__()
        T, C, _, _ = shape_in
        model_type = normalize_simvp_model_type(model_type)
        self.model_type = model_type

        if model_type == "incepu":
            self.enc = Encoder(C, hid_S, N_S)
            self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
            self.dec = Decoder(hid_S, C, N_S)
        elif model_type == "gsta":
            act_inplace = False
            self.enc = MetaEncoder(
                C,
                hid_S,
                N_S,
                spatio_kernel=spatio_kernel_enc,
                act_inplace=act_inplace,
            )
            self.hid = MidMetaNet(
                T * hid_S,
                hid_T,
                N_T,
                model_type=model_type,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
            )
            self.dec = MetaDecoder(
                hid_S,
                C,
                N_S,
                spatio_kernel=spatio_kernel_dec,
                act_inplace=act_inplace,
            )
        else:
            raise ValueError(
                f"Unsupported SimVP model_type '{model_type}'. Available choices: ('incepu', 'gsta')."
            )

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.reshape(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y
