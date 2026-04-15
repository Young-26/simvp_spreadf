import torch
import torch.nn as nn

from .convlstm_model import _parse_num_hidden, reshape_patch, reshape_patch_back


class CausalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf PredRNN++ currently only supports predrnnpp_stride=1. "
                "stride>1 changes hidden/memory spatial sizes and LayerNorm assumptions, "
                "but this integration keeps the OpenSTL-style same-resolution state update."
            )

        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 7, height, width]),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 4, height, width]),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 3, height, width]),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 3, height, width]),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden, height, width]),
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 4, height, width]),
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden, height, width]),
            )
        else:
            self.conv_x = nn.Conv2d(
                in_channel,
                num_hidden * 7,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_h = nn.Conv2d(
                num_hidden,
                num_hidden * 4,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_c = nn.Conv2d(
                num_hidden,
                num_hidden * 3,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_m = nn.Conv2d(
                num_hidden,
                num_hidden * 3,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_o = nn.Conv2d(
                num_hidden * 2,
                num_hidden,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_c2m = nn.Conv2d(
                num_hidden,
                num_hidden * 4,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_om = nn.Conv2d(
                num_hidden,
                num_hidden,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        self.conv_last = nn.Conv2d(
            num_hidden * 2,
            num_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_concat = self.conv_c(c_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat,
            self.num_hidden,
            dim=1,
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)
        c_new = f_t * c_t + i_t * g_t

        c2m = self.conv_c2m(c_new)
        i_c2m, g_c2m, f_c2m, o_c = torch.split(c2m, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_m + i_c2m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + f_c2m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c2m)
        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime

        o_m = self.conv_om(m_new)
        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        mem = torch.cat((c_new, m_new), dim=1)
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf PredRNN++ currently only supports predrnnpp_stride=1. "
                "stride>1 changes hidden/highway spatial sizes and LayerNorm assumptions, "
                "but this integration keeps the OpenSTL-style same-resolution state update."
            )

        self.num_hidden = num_hidden
        padding = filter_size // 2

        if layer_norm:
            self.z_concat = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 2,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 2, height, width]),
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 2,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 2, height, width]),
            )
        else:
            self.z_concat = nn.Conv2d(
                in_channel,
                num_hidden * 2,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.x_concat = nn.Conv2d(
                in_channel,
                num_hidden * 2,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

    def forward(self, x, z):
        if z is None:
            z = torch.zeros_like(x)

        gates = self.x_concat(x) + self.z_concat(z)
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        return u * p + (1.0 - u) * z


class PredRNNpp_Model(nn.Module):
    r"""PredRNN++ backbone adapted from OpenSTL for direct multi-step forecasting."""

    def __init__(
        self,
        shape_in,
        out_T,
        num_hidden="128,128,128,128",
        filter_size=5,
        patch_size=4,
        stride=1,
        layer_norm=False,
    ):
        super().__init__()
        in_T, channels, height, width = shape_in

        self.in_T = in_T
        self.out_T = out_T
        self.input_channels = channels
        self.patch_size = patch_size
        self.num_hidden = _parse_num_hidden(num_hidden)
        self.num_layers = len(self.num_hidden)
        self.frame_channel = patch_size * patch_size * channels

        if self.num_layers < 2:
            raise ValueError("PredRNN++ requires at least two hidden layers.")
        if stride != 1:
            raise ValueError(
                "simvp_spreadf PredRNN++ currently only supports predrnnpp_stride=1. "
                "The migrated OpenSTL cell stack assumes same-resolution hidden, cell, and memory states."
            )

        patch_h = height // patch_size
        patch_w = width // patch_size
        if patch_h * patch_size != height or patch_w * patch_size != width:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch_size={patch_size}."
            )

        self.gradient_highway = GHU(
            in_channel=self.num_hidden[0],
            num_hidden=self.num_hidden[0],
            height=patch_h,
            width=patch_w,
            filter_size=filter_size,
            stride=stride,
            layer_norm=layer_norm,
        )

        cell_list = []
        for idx in range(self.num_layers):
            in_channel = self.frame_channel if idx == 0 else self.num_hidden[idx - 1]
            cell_list.append(
                CausalLSTMCell(
                    in_channel=in_channel,
                    num_hidden=self.num_hidden[idx],
                    height=patch_h,
                    width=patch_w,
                    filter_size=filter_size,
                    stride=stride,
                    layer_norm=layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            self.num_hidden[-1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, frames_tensor):
        if frames_tensor.ndim != 5:
            raise ValueError(f"PredRNN++ expects [B, T, C, H, W], but got {frames_tensor.shape}.")

        batch, in_steps, channels, _, _ = frames_tensor.shape
        if channels != self.input_channels:
            raise ValueError(
                f"PredRNN++ was built for {self.input_channels} input channels, but got {channels}."
            )
        if in_steps != self.in_T:
            raise ValueError(
                f"PredRNN++ was built for in_T={self.in_T}, but received sequence length {in_steps}."
            )

        frames = reshape_patch(frames_tensor, self.patch_size)
        patch_h, patch_w = frames.shape[-2:]

        h_t = []
        c_t = []
        for hidden in self.num_hidden:
            zeros = frames.new_zeros(batch, hidden, patch_h, patch_w)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = frames.new_zeros(batch, self.num_hidden[0], patch_h, patch_w)
        z_t = None
        x_gen = None
        next_frames = []
        total_steps = in_steps + self.out_T - 1

        # Mirror the existing simvp_spreadf ConvLSTM integration:
        # observed frames use teacher forcing, forecast frames roll out autoregressively.
        for t in range(total_steps):
            net = frames[:, t] if t < in_steps else x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1], c_t[1], memory)

            for layer_idx in range(2, self.num_layers):
                h_t[layer_idx], c_t[layer_idx], memory = self.cell_list[layer_idx](
                    h_t[layer_idx - 1],
                    h_t[layer_idx],
                    c_t[layer_idx],
                    memory,
                )

            x_gen = self.conv_last(h_t[-1])
            if t >= in_steps - 1:
                next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        return reshape_patch_back(next_frames, self.patch_size, self.input_channels)
