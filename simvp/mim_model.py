import torch
import torch.nn as nn

from .convlstm_model import _parse_num_hidden, reshape_patch, reshape_patch_back


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf MIM currently only supports mim_stride=1. "
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

        # Kept for structural parity with OpenSTL's reference MIMN module. The original code
        # registers this projection but does not consume it in forward(), so DDP must tolerate
        # unused parameters when training MIM on multiple GPUs.
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
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat,
            self.num_hidden,
            dim=1,
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class MIMBlock(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf MIM currently only supports mim_stride=1. "
                "stride>1 changes hidden/memory spatial sizes and LayerNorm assumptions, "
                "but this integration keeps the OpenSTL-style same-resolution state update."
            )

        self.convlstm_c = None
        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        if layer_norm:
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 3, height, width]),
            )
            self.conv_s_cc = nn.Sequential(
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
            self.conv_x_cc = nn.Sequential(
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
            self.conv_h_concat = nn.Sequential(
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
            self.conv_x_concat = nn.Sequential(
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
            self.conv_t_cc = nn.Conv2d(
                in_channel,
                num_hidden * 3,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_s_cc = nn.Conv2d(
                num_hidden,
                num_hidden * 4,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_x_cc = nn.Conv2d(
                num_hidden,
                num_hidden * 4,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_h_concat = nn.Conv2d(
                num_hidden,
                num_hidden * 4,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_x_concat = nn.Conv2d(
                num_hidden,
                num_hidden * 4,
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

    def reset_state(self):
        self.convlstm_c = None

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def _mims(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        h = self._init_state(x) if h is None else h
        c = self._init_state(x) if c is None else c
        m = self._init_state(x) if m is None else m
        diff_h = self._init_state(x) if diff_h is None else diff_h

        t_cc = self.conv_t_cc(h)
        s_cc = self.conv_s_cc(m)
        x_cc = self.conv_x_cc(x)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_prime = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_prime = torch.tanh(g_x + g_s)
        f_prime = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_prime * m + i_prime * g_prime

        c, self.convlstm_c = self._mims(
            diff_h,
            c,
            self.convlstm_c if self.convlstm_c is None else self.convlstm_c.detach(),
        )

        new_c = c + i * g
        cell = torch.cat((new_c, new_m), dim=1)
        new_h = o * torch.tanh(self.conv_last(cell))
        return new_h, new_c, new_m


class MIMN(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf MIM currently only supports mim_stride=1. "
                "stride>1 changes hidden/memory spatial sizes and LayerNorm assumptions, "
                "but this integration keeps the OpenSTL-style same-resolution state update."
            )

        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        if layer_norm:
            self.conv_h_concat = nn.Sequential(
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
            self.conv_x_concat = nn.Sequential(
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
        else:
            self.conv_h_concat = nn.Conv2d(
                in_channel,
                num_hidden * 4,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv_x_concat = nn.Conv2d(
                in_channel,
                num_hidden * 4,
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

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)
        return h_new, c_new


class MIM_Model(nn.Module):
    r"""MIM backbone adapted from OpenSTL for direct multi-step forecasting."""

    def __init__(
        self,
        shape_in,
        out_T,
        num_hidden="128,128,128,128",
        filter_size=5,
        patch_size=4,
        stride=1,
        layer_norm=False,
        reverse_scheduled_sampling=False,
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
        self.reverse_scheduled_sampling = bool(reverse_scheduled_sampling)
        self.mse_criterion = nn.MSELoss()
        # This port keeps the OpenSTL MIM structure intact. In that structure the shared
        # spatiotemporal memory path and the MIMBlock/MIMN internal state bootstrapping are
        # only shape-safe when every recurrent layer uses the same hidden width.
        if len(set(self.num_hidden)) != 1:
            raise ValueError(
                "MIM currently requires identical hidden sizes across all layers because the "
                "OpenSTL-style shared memory path and MIMBlock/MIMN internal state initialization "
                "assume one common hidden width. Received num_hidden="
                f"{self.num_hidden}. Use values like '128,128,128,128', not heterogeneous lists."
            )

        if stride != 1:
            raise ValueError(
                "simvp_spreadf MIM currently only supports mim_stride=1. "
                "The migrated OpenSTL cell stack assumes same-resolution hidden, cell, and memory states."
            )

        patch_h = height // patch_size
        patch_w = width // patch_size
        if patch_h * patch_size != height or patch_w * patch_size != width:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch_size={patch_size}."
            )

        stlstm_layers = []
        diff_layers = []
        for idx in range(self.num_layers):
            in_channel = self.frame_channel if idx == 0 else self.num_hidden[idx - 1]
            if idx == 0:
                stlstm_layers.append(
                    SpatioTemporalLSTMCell(
                        in_channel=in_channel,
                        num_hidden=self.num_hidden[idx],
                        height=patch_h,
                        width=patch_w,
                        filter_size=filter_size,
                        stride=stride,
                        layer_norm=layer_norm,
                    )
                )
            else:
                stlstm_layers.append(
                    MIMBlock(
                        in_channel=in_channel,
                        num_hidden=self.num_hidden[idx],
                        height=patch_h,
                        width=patch_w,
                        filter_size=filter_size,
                        stride=stride,
                        layer_norm=layer_norm,
                    )
                )

        for idx in range(self.num_layers - 1):
            diff_layers.append(
                MIMN(
                    in_channel=self.num_hidden[idx],
                    num_hidden=self.num_hidden[idx + 1],
                    height=patch_h,
                    width=patch_w,
                    filter_size=filter_size,
                    stride=stride,
                    layer_norm=layer_norm,
                )
            )

        self.stlstm_layers = nn.ModuleList(stlstm_layers)
        self.diff_layers = nn.ModuleList(diff_layers)
        self.conv_last = nn.Conv2d(
            self.num_hidden[-1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def _mask_to_channel_first(self, mask_true, expected_steps, patch_h, patch_w):
        if mask_true is None:
            return None
        if mask_true.ndim != 5:
            raise ValueError(f"MIM mask_true expects 5 dims, but got {mask_true.shape}.")
        if mask_true.shape[1] != expected_steps:
            raise ValueError(
                f"MIM mask_true expects time length {expected_steps}, but got {mask_true.shape[1]}."
            )

        if (
            mask_true.shape[2] == self.frame_channel
            and mask_true.shape[3] == patch_h
            and mask_true.shape[4] == patch_w
        ):
            return mask_true.contiguous()
        if (
            mask_true.shape[2] == patch_h
            and mask_true.shape[3] == patch_w
            and mask_true.shape[4] == self.frame_channel
        ):
            return mask_true.permute(0, 1, 4, 2, 3).contiguous()

        raise ValueError(
            "MIM mask_true must be channel-first [B, T, Cp, Hp, Wp] or "
            "channel-last [B, T, Hp, Wp, Cp]. "
            f"Got {mask_true.shape} with expected patch grid {(patch_h, patch_w)} and channels {self.frame_channel}."
        )

    def _resolve_openstl_loss_pair(self, next_frames, frames_tensor, loss_target):
        if loss_target is None:
            if frames_tensor.shape[1] == self.in_T + self.out_T:
                return next_frames, frames_tensor[:, 1:]
            return None, None

        if loss_target.ndim != 5:
            raise ValueError(f"MIM loss_target expects [B, T, C, H, W], but got {loss_target.shape}.")
        if loss_target.shape[0] != next_frames.shape[0] or loss_target.shape[2:] != next_frames.shape[2:]:
            raise ValueError(
                f"MIM loss_target shape {loss_target.shape} is incompatible with predictions {next_frames.shape}."
            )

        if loss_target.shape[1] == next_frames.shape[1]:
            return next_frames, loss_target
        if loss_target.shape[1] == self.out_T:
            return next_frames[:, -self.out_T :], loss_target

        raise ValueError(
            f"MIM loss_target time length must be {next_frames.shape[1]} or {self.out_T}, "
            f"but got {loss_target.shape[1]}."
        )

    def forward(
        self,
        frames_tensor,
        mask_true=None,
        return_loss=False,
        loss_target=None,
    ):
        if frames_tensor.ndim != 5:
            raise ValueError(f"MIM expects [B, T, C, H, W], but got {frames_tensor.shape}.")

        batch, in_steps, channels, _, _ = frames_tensor.shape
        if channels != self.input_channels:
            raise ValueError(
                f"MIM was built for {self.input_channels} input channels, but got {channels}."
            )

        full_sequence_length = self.in_T + self.out_T
        if in_steps not in (self.in_T, full_sequence_length):
            raise ValueError(
                "MIM expects either the observed sequence length "
                f"{self.in_T} or the full training sequence length {full_sequence_length}, but got {in_steps}."
            )

        frames = reshape_patch(frames_tensor, self.patch_size)
        patch_h, patch_w = frames.shape[-2:]

        if self.reverse_scheduled_sampling:
            expected_mask_steps = max(full_sequence_length - 2, 0) if in_steps == full_sequence_length else 0
        else:
            expected_mask_steps = max(self.out_T - 1, 0) if in_steps == full_sequence_length else 0
        mask_true = self._mask_to_channel_first(mask_true, expected_mask_steps, patch_h, patch_w)

        total_steps = full_sequence_length - 1 if in_steps == full_sequence_length else self.in_T + self.out_T - 1
        collect_all_steps = in_steps == full_sequence_length

        for layer in self.stlstm_layers[1:]:
            if isinstance(layer, MIMBlock):
                layer.reset_state()

        h_t = []
        c_t = []
        hidden_state_diff = []
        cell_state_diff = []
        for hidden in self.num_hidden:
            zeros = frames.new_zeros(batch, hidden, patch_h, patch_w)
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)

        st_memory = frames.new_zeros(batch, self.num_hidden[0], patch_h, patch_w)
        x_gen = None
        next_frames = []

        for t in range(total_steps):
            if self.reverse_scheduled_sampling:
                if t == 0:
                    net = frames[:, 0]
                elif mask_true is None:
                    net = frames[:, t] if t < in_steps else x_gen
                elif t < in_steps:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
                else:
                    net = x_gen
            else:
                if t < self.in_T:
                    net = frames[:, t]
                elif mask_true is not None and t < in_steps:
                    mask_idx = t - self.in_T
                    net = mask_true[:, mask_idx] * frames[:, t] + (1 - mask_true[:, mask_idx]) * x_gen
                else:
                    net = x_gen

            prev_h0 = h_t[0]
            h_t[0], c_t[0], st_memory = self.stlstm_layers[0](net, h_t[0], c_t[0], st_memory)

            for layer_idx in range(1, self.num_layers):
                if t > 0:
                    if layer_idx == 1:
                        hidden_state_diff[layer_idx - 1], cell_state_diff[layer_idx - 1] = self.diff_layers[
                            layer_idx - 1
                        ](
                            h_t[layer_idx - 1] - prev_h0,
                            hidden_state_diff[layer_idx - 1],
                            cell_state_diff[layer_idx - 1],
                        )
                    else:
                        hidden_state_diff[layer_idx - 1], cell_state_diff[layer_idx - 1] = self.diff_layers[
                            layer_idx - 1
                        ](
                            hidden_state_diff[layer_idx - 2],
                            hidden_state_diff[layer_idx - 1],
                            cell_state_diff[layer_idx - 1],
                        )

                h_t[layer_idx], c_t[layer_idx], st_memory = self.stlstm_layers[layer_idx](
                    h_t[layer_idx - 1],
                    hidden_state_diff[layer_idx - 1],
                    h_t[layer_idx],
                    c_t[layer_idx],
                    st_memory,
                )

            x_gen = self.conv_last(h_t[-1])
            if collect_all_steps or t >= self.in_T - 1:
                next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        next_frames = reshape_patch_back(next_frames, self.patch_size, self.input_channels)

        if return_loss:
            loss_pred, loss_target = self._resolve_openstl_loss_pair(next_frames, frames_tensor, loss_target)
            if loss_pred is None or loss_target is None:
                raise ValueError(
                    "MIM needs either a full [x, y] sequence input or an explicit loss_target "
                    "when return_loss=True."
                )
            loss = self.mse_criterion(loss_pred, loss_target)
            return next_frames, loss

        return next_frames
