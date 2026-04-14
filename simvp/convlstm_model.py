import torch
import torch.nn as nn


def _parse_num_hidden(num_hidden):
    if isinstance(num_hidden, str):
        values = [int(x.strip()) for x in num_hidden.split(",") if x.strip()]
    else:
        values = [int(x) for x in num_hidden]
    if not values:
        raise ValueError("ConvLSTM requires at least one hidden dimension.")
    return values


def reshape_patch(frames, patch_size):
    if patch_size == 1:
        return frames

    batch, steps, channels, height, width = frames.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Input size {(height, width)} must be divisible by patch_size={patch_size}."
        )

    patch_h = height // patch_size
    patch_w = width // patch_size
    frames = frames.reshape(
        batch,
        steps,
        channels,
        patch_h,
        patch_size,
        patch_w,
        patch_size,
    )
    frames = frames.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
    frames = frames.reshape(batch, steps, patch_h, patch_w, patch_size * patch_size * channels)
    return frames.permute(0, 1, 4, 2, 3).contiguous()


def reshape_patch_back(patch_frames, patch_size, out_channels):
    if patch_size == 1:
        return patch_frames

    batch, steps, channels, patch_h, patch_w = patch_frames.shape
    expected_channels = out_channels * patch_size * patch_size
    if channels != expected_channels:
        raise ValueError(
            f"Patch tensor channel count {channels} does not match expected {expected_channels}."
        )

    patch_frames = patch_frames.permute(0, 1, 3, 4, 2).contiguous()
    patch_frames = patch_frames.reshape(
        batch,
        steps,
        patch_h,
        patch_w,
        patch_size,
        patch_size,
        out_channels,
    )
    patch_frames = patch_frames.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
    return patch_frames.reshape(
        batch,
        steps,
        out_channels,
        patch_h * patch_size,
        patch_w * patch_size,
    )


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        self.num_hidden = num_hidden
        padding = filter_size // 2
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 4, height, width]),
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
        else:
            self.conv_x = nn.Conv2d(
                in_channel,
                num_hidden * 4,
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

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM backbone adapted from OpenSTL for direct multi-step forecasting."""

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

        patch_h = height // patch_size
        patch_w = width // patch_size
        if patch_h * patch_size != height or patch_w * patch_size != width:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch_size={patch_size}."
            )

        cell_list = []
        for idx in range(self.num_layers):
            in_channel = self.frame_channel if idx == 0 else self.num_hidden[idx - 1]
            cell_list.append(
                ConvLSTMCell(
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
            raise ValueError(f"ConvLSTM expects [B, T, C, H, W], but got {frames_tensor.shape}.")

        batch, in_steps, channels, height, width = frames_tensor.shape
        if channels != self.input_channels:
            raise ValueError(
                f"ConvLSTM was built for {self.input_channels} input channels, but got {channels}."
            )
        if in_steps != self.in_T:
            raise ValueError(
                f"ConvLSTM was built for in_T={self.in_T}, but received sequence length {in_steps}."
            )

        frames = reshape_patch(frames_tensor, self.patch_size)
        patch_h, patch_w = frames.shape[-2:]

        h_t = []
        c_t = []
        for hidden in self.num_hidden:
            zeros = frames.new_zeros(batch, hidden, patch_h, patch_w)
            h_t.append(zeros)
            c_t.append(zeros)

        next_frames = []
        x_gen = None
        total_steps = in_steps + self.out_T - 1

        for t in range(total_steps):
            net = frames[:, t] if t < in_steps else x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
            for layer_idx in range(1, self.num_layers):
                h_t[layer_idx], c_t[layer_idx] = self.cell_list[layer_idx](
                    h_t[layer_idx - 1],
                    h_t[layer_idx],
                    c_t[layer_idx],
                )

            x_gen = self.conv_last(h_t[-1])
            if t >= in_steps - 1:
                next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        return reshape_patch_back(next_frames, self.patch_size, self.input_channels)
