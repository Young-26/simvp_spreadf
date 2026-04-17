import math

import torch
from torch import nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(
            x,
            drop_prob=self.drop_prob,
            training=self.training,
            scale_by_keep=self.scale_by_keep,
        )


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        upsampling: bool = False,
        act_norm: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        self.act_norm = act_norm

        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
        act_norm: bool = True,
        act_inplace: bool = True,
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsampling=upsampling,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def sampling_generator(num_layers: int, reverse: bool = False):
    samplings = [False, True] * ((num_layers + 1) // 2)
    samplings = samplings[:num_layers]
    if reverse:
        return list(reversed(samplings))
    return samplings


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, spatio_kernel: int, act_inplace: bool):
        super().__init__()
        samplings = sampling_generator(num_layers)
        self.enc = nn.Sequential(
            ConvSC(
                in_channels,
                hidden_channels,
                spatio_kernel,
                downsampling=samplings[0],
                act_inplace=act_inplace,
            ),
            *[
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    spatio_kernel,
                    downsampling=sampling,
                    act_inplace=act_inplace,
                )
                for sampling in samplings[1:]
            ],
        )

    def forward(self, x: torch.Tensor):
        enc1 = self.enc[0](x)
        latent = enc1
        for layer in self.enc[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, spatio_kernel: int, act_inplace: bool):
        super().__init__()
        samplings = sampling_generator(num_layers, reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    spatio_kernel,
                    upsampling=sampling,
                    act_inplace=act_inplace,
                )
                for sampling in samplings
            ]
        )
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, hidden: torch.Tensor, enc1: torch.Tensor) -> torch.Tensor:
        for layer in self.dec[:-1]:
            hidden = layer(hidden)
        y = self.dec[-1](hidden + enc1)
        return self.readout(y)


class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class MixMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalAttentionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 3, reduction: int = 16):
        super().__init__()
        dilated_kernel = 2 * dilation - 1
        dilated_padding = (dilated_kernel - 1) // 2
        large_kernel = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        large_padding = dilation * (large_kernel - 1) // 2

        self.conv0 = nn.Conv2d(dim, dim, dilated_kernel, padding=dilated_padding, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            large_kernel,
            stride=1,
            padding=large_padding,
            groups=dim,
            dilation=dilation,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

        reduction_channels = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduction_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        features = self.conv1(attn)

        batch_size, channels = x.shape[:2]
        se_attn = self.avg_pool(x).view(batch_size, channels)
        se_attn = self.fc(se_attn).view(batch_size, channels, 1, 1)
        return se_attn * features * residual


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 21, attn_shortcut: bool = True):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = TemporalAttentionModule(dim, kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone() if self.attn_shortcut else None
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if shortcut is not None:
            x = x + shortcut
        return x


class TAUSubBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        init_value: float = 1e-2,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = TemporalAttention(dim, kernel_size)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = TAUSubBlock(
            in_channels,
            kernel_size=21,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
        )
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.reduction = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block(x)
        if self.reduction is not None:
            z = self.reduction(z)
        return z


class MidMetaNet(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        num_layers: int,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"TAU requires N_T >= 2, but got {num_layers}.")
        self.num_layers = num_layers
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.num_layers)]

        layers = [
            MetaBlock(
                channel_in,
                channel_hid,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=dpr[0],
            )
        ]
        for layer_idx in range(1, num_layers - 1):
            layers.append(
                MetaBlock(
                    channel_hid,
                    channel_hid,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[layer_idx],
                )
            )
        layers.append(
            MetaBlock(
                channel_hid,
                channel_in,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
            )
        )
        self.enc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, height, width = x.shape
        z = x.reshape(batch_size, steps * channels, height, width)
        for layer in self.enc:
            z = layer(z)
        return z.reshape(batch_size, steps, channels, height, width)


class TAU_Model(nn.Module):
    def __init__(
        self,
        shape_in,
        hid_S: int = 16,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
    ):
        super().__init__()
        steps, channels, height, width = shape_in
        act_inplace = False

        self.enc = Encoder(
            channels,
            hid_S,
            N_S,
            spatio_kernel=spatio_kernel_enc,
            act_inplace=act_inplace,
        )
        self.dec = Decoder(
            hid_S,
            channels,
            N_S,
            spatio_kernel=spatio_kernel_dec,
            act_inplace=act_inplace,
        )
        self.hid = MidMetaNet(
            steps * hid_S,
            hid_T,
            N_T,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, height, width = x_raw.shape
        x = x_raw.reshape(batch_size * steps, channels, height, width)

        embed, skip = self.enc(x)
        _, hidden_channels, hidden_height, hidden_width = embed.shape

        z = embed.reshape(batch_size, steps, hidden_channels, hidden_height, hidden_width)
        hidden = self.hid(z)
        hidden = hidden.reshape(batch_size * steps, hidden_channels, hidden_height, hidden_width)

        y = self.dec(hidden, skip)
        return y.reshape(batch_size, steps, channels, height, width)


def tau_diff_div_reg(pred_y: torch.Tensor, batch_y: torch.Tensor, tau: float = 0.1, eps: float = 1e-12) -> torch.Tensor:
    batch_size, steps, channels = pred_y.shape[:3]
    del channels
    if steps <= 2:
        return pred_y.new_zeros(())

    gap_pred = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(batch_size, steps - 1, -1)
    gap_true = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(batch_size, steps - 1, -1)

    softmax_gap_pred = torch.softmax(gap_pred / tau, dim=-1)
    softmax_gap_true = torch.softmax(gap_true / tau, dim=-1)
    loss_gap = softmax_gap_pred * torch.log(softmax_gap_pred / (softmax_gap_true + eps) + eps)
    return loss_gap.mean()
