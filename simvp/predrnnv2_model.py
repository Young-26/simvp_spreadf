import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm_model import _parse_num_hidden, reshape_patch, reshape_patch_back


class SpatioTemporalLSTMCellv2(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf PredRNNv2 currently only supports predrnnv2_stride=1. "
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
        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new, delta_c, delta_m


class PredRNNv2_Model(nn.Module):
    r"""PredRNNv2 backbone adapted from OpenSTL for direct multi-step forecasting."""

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
        decouple_beta=0.1,
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
        self.decouple_beta = float(decouple_beta)
        self.mse_criterion = nn.MSELoss()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf PredRNNv2 currently only supports predrnnv2_stride=1. "
                "The migrated OpenSTL cell stack assumes same-resolution hidden, cell, and memory states."
            )

        patch_h = height // patch_size
        patch_w = width // patch_size
        if patch_h * patch_size != height or patch_w * patch_size != width:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch_size={patch_size}."
            )
        if len(set(self.num_hidden)) != 1:
            raise ValueError(
                "PredRNNv2 currently expects identical hidden sizes across layers because the "
                "OpenSTL decoupling adapter is shared across the recurrent stack."
            )

        cell_list = []
        for idx in range(self.num_layers):
            in_channel = self.frame_channel if idx == 0 else self.num_hidden[idx - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(
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
        adapter_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_hidden,
            adapter_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def _mask_to_channel_first(self, mask_true, expected_steps, patch_h, patch_w):
        if mask_true is None:
            return None
        if mask_true.ndim != 5:
            raise ValueError(f"PredRNNv2 mask_true expects 5 dims, but got {mask_true.shape}.")
        if mask_true.shape[1] != expected_steps:
            raise ValueError(
                f"PredRNNv2 mask_true expects time length {expected_steps}, but got {mask_true.shape[1]}."
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
            "PredRNNv2 mask_true must be channel-first [B, T, Cp, Hp, Wp] or "
            "channel-last [B, T, Hp, Wp, Cp]. "
            f"Got {mask_true.shape} with expected patch grid {(patch_h, patch_w)} and channels {self.frame_channel}."
        )

    def _resolve_openstl_loss_pair(self, next_frames, frames_tensor, loss_target):
        if loss_target is None:
            if frames_tensor.shape[1] == self.in_T + self.out_T:
                return next_frames, frames_tensor[:, 1:]
            return None, None

        if loss_target.ndim != 5:
            raise ValueError(f"PredRNNv2 loss_target expects [B, T, C, H, W], but got {loss_target.shape}.")
        if loss_target.shape[0] != next_frames.shape[0] or loss_target.shape[2:] != next_frames.shape[2:]:
            raise ValueError(
                f"PredRNNv2 loss_target shape {loss_target.shape} is incompatible with predictions {next_frames.shape}."
            )

        if loss_target.shape[1] == next_frames.shape[1]:
            return next_frames, loss_target
        if loss_target.shape[1] == self.out_T:
            return next_frames[:, -self.out_T :], loss_target

        raise ValueError(
            f"PredRNNv2 loss_target time length must be {next_frames.shape[1]} or {self.out_T}, "
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
            raise ValueError(f"PredRNNv2 expects [B, T, C, H, W], but got {frames_tensor.shape}.")

        batch, in_steps, channels, _, _ = frames_tensor.shape
        if channels != self.input_channels:
            raise ValueError(
                f"PredRNNv2 was built for {self.input_channels} input channels, but got {channels}."
            )

        full_sequence_length = self.in_T + self.out_T
        if in_steps not in (self.in_T, full_sequence_length):
            raise ValueError(
                "PredRNNv2 expects either the observed sequence length "
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

        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        for hidden in self.num_hidden:
            zeros = frames.new_zeros(batch, hidden, patch_h, patch_w)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = frames.new_zeros(batch, self.num_hidden[0], patch_h, patch_w)
        x_gen = None
        next_frames = []
        decouple_terms = []

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

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).reshape(delta_c.shape[0], delta_c.shape[1], -1),
                dim=2,
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).reshape(delta_m.shape[0], delta_m.shape[1], -1),
                dim=2,
            )

            for layer_idx in range(1, self.num_layers):
                h_t[layer_idx], c_t[layer_idx], memory, delta_c, delta_m = self.cell_list[layer_idx](
                    h_t[layer_idx - 1],
                    h_t[layer_idx],
                    c_t[layer_idx],
                    memory,
                )
                delta_c_list[layer_idx] = F.normalize(
                    self.adapter(delta_c).reshape(delta_c.shape[0], delta_c.shape[1], -1),
                    dim=2,
                )
                delta_m_list[layer_idx] = F.normalize(
                    self.adapter(delta_m).reshape(delta_m.shape[0], delta_m.shape[1], -1),
                    dim=2,
                )

            x_gen = self.conv_last(h_t[-1])
            if collect_all_steps or t >= self.in_T - 1:
                next_frames.append(x_gen)

            if return_loss:
                for layer_idx in range(self.num_layers):
                    decouple_terms.append(
                        torch.mean(
                            torch.abs(
                                torch.cosine_similarity(
                                    delta_c_list[layer_idx],
                                    delta_m_list[layer_idx],
                                    dim=2,
                                )
                            )
                        )
                    )

        next_frames = torch.stack(next_frames, dim=1)
        next_frames = reshape_patch_back(next_frames, self.patch_size, self.input_channels)

        if return_loss:
            loss_pred, loss_target = self._resolve_openstl_loss_pair(next_frames, frames_tensor, loss_target)
            if loss_pred is None or loss_target is None:
                raise ValueError(
                    "PredRNNv2 needs either a full [x, y] sequence input or an explicit loss_target "
                    "when return_loss=True."
                )
            decouple_loss = (
                torch.mean(torch.stack(decouple_terms, dim=0))
                if decouple_terms
                else next_frames.new_zeros(())
            )
            loss = self.mse_criterion(loss_pred, loss_target) + self.decouple_beta * decouple_loss
            return next_frames, loss

        return next_frames
