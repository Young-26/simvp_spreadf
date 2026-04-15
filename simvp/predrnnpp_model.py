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
    def __init__(
        self,
        in_channel,
        num_hidden,
        height,
        width,
        filter_size,
        stride,
        layer_norm,
        initializer=0.001,
    ):
        super().__init__()

        if stride != 1:
            raise ValueError(
                "simvp_spreadf PredRNN++ currently only supports predrnnpp_stride=1. "
                "stride>1 changes hidden/highway spatial sizes and LayerNorm assumptions, "
                "but this integration keeps the OpenSTL-style same-resolution state update."
            )

        self.num_hidden = num_hidden
        self.initializer = initializer
        padding = filter_size // 2

        if layer_norm:
            # OpenSTL's reference code uses LayerNorm([num_hidden, H, W]) after a Conv2d whose
            # output channel count is 2 * num_hidden. In standard PyTorch LayerNorm the
            # normalized_shape must match the trailing dimensions exactly, so we normalize the
            # actual Conv2d output shape before splitting it into (p, u) halves.
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

        if initializer != -1:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.uniform_(module.weight, -self.initializer, self.initializer)

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
        reverse_scheduled_sampling=False,
        ghu_initializer=0.001,
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
            initializer=ghu_initializer,
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

    def _normalize_recipe(self, recipe: str) -> str:
        recipe = str(recipe).lower()
        if recipe not in {"simvp", "openstl"}:
            raise ValueError(f"Unsupported PredRNN++ recipe '{recipe}'.")
        return recipe

    def _mask_to_channel_first(self, mask_true, expected_steps, patch_h, patch_w):
        if mask_true is None:
            return None
        if mask_true.ndim != 5:
            raise ValueError(f"PredRNN++ mask_true expects 5 dims, but got {mask_true.shape}.")
        if mask_true.shape[1] != expected_steps:
            raise ValueError(
                f"PredRNN++ mask_true expects time length {expected_steps}, but got {mask_true.shape[1]}."
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
            "PredRNN++ mask_true must be channel-first [B, T, Cp, Hp, Wp] or "
            "channel-last [B, T, Hp, Wp, Cp]. "
            f"Got {mask_true.shape} with expected patch grid {(patch_h, patch_w)} and channels {self.frame_channel}."
        )

    def _resolve_openstl_loss_pair(self, next_frames, frames_tensor, loss_target):
        if loss_target is None:
            if frames_tensor.shape[1] == self.in_T + self.out_T:
                return next_frames, frames_tensor[:, 1:]
            return None, None

        if loss_target.ndim != 5:
            raise ValueError(f"PredRNN++ loss_target expects [B, T, C, H, W], but got {loss_target.shape}.")
        if loss_target.shape[0] != next_frames.shape[0] or loss_target.shape[2:] != next_frames.shape[2:]:
            raise ValueError(
                f"PredRNN++ loss_target shape {loss_target.shape} is incompatible with predictions {next_frames.shape}."
            )

        if loss_target.shape[1] == next_frames.shape[1]:
            return next_frames, loss_target
        if loss_target.shape[1] == self.out_T:
            return next_frames[:, -self.out_T :], loss_target

        raise ValueError(
            f"PredRNN++ loss_target time length must be {next_frames.shape[1]} or {self.out_T}, "
            f"but got {loss_target.shape[1]}."
        )

    def forward(
        self,
        frames_tensor,
        mask_true=None,
        return_loss=False,
        loss_target=None,
        recipe="simvp",
    ):
        recipe = self._normalize_recipe(recipe)
        if frames_tensor.ndim != 5:
            raise ValueError(f"PredRNN++ expects [B, T, C, H, W], but got {frames_tensor.shape}.")

        batch, in_steps, channels, _, _ = frames_tensor.shape
        if channels != self.input_channels:
            raise ValueError(
                f"PredRNN++ was built for {self.input_channels} input channels, but got {channels}."
            )

        full_sequence_length = self.in_T + self.out_T
        if recipe == "simvp":
            if in_steps != self.in_T:
                raise ValueError(
                    f"PredRNN++ recipe='simvp' was built for in_T={self.in_T}, but received length {in_steps}."
                )
        elif in_steps not in (self.in_T, full_sequence_length):
            raise ValueError(
                "PredRNN++ recipe='openstl' expects either the observed sequence length "
                f"{self.in_T} or the full training sequence length {full_sequence_length}, but got {in_steps}."
            )

        frames = reshape_patch(frames_tensor, self.patch_size)
        patch_h, patch_w = frames.shape[-2:]

        if recipe == "openstl":
            if self.reverse_scheduled_sampling:
                expected_mask_steps = max(full_sequence_length - 2, 0) if in_steps == full_sequence_length else 0
            else:
                expected_mask_steps = max(self.out_T - 1, 0) if in_steps == full_sequence_length else 0
            mask_true = self._mask_to_channel_first(mask_true, expected_mask_steps, patch_h, patch_w)
            total_steps = full_sequence_length - 1 if in_steps == full_sequence_length else self.in_T + self.out_T - 1
            collect_all_steps = in_steps == full_sequence_length
        else:
            total_steps = in_steps + self.out_T - 1
            collect_all_steps = False

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

        for t in range(total_steps):
            if recipe == "simvp":
                net = frames[:, t] if t < in_steps else x_gen
            elif self.reverse_scheduled_sampling:
                if t == 0:
                    net = frames[:, 0]
                elif mask_true is None:
                    net = frames[:, t] if t < in_steps else x_gen
                elif mask_true is not None and t < in_steps:
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
            if collect_all_steps or t >= self.in_T - 1:
                next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        next_frames = reshape_patch_back(next_frames, self.patch_size, self.input_channels)

        if return_loss:
            if recipe == "openstl":
                loss_pred, loss_target = self._resolve_openstl_loss_pair(next_frames, frames_tensor, loss_target)
                if loss_pred is None or loss_target is None:
                    raise ValueError(
                        "PredRNN++ recipe='openstl' needs either a full [x, y] sequence input or an explicit loss_target "
                        "when return_loss=True."
                    )
                loss = self.mse_criterion(loss_pred, loss_target)
            else:
                if loss_target is None:
                    raise ValueError(
                        "PredRNN++ recipe='simvp' does not define a default loss target. "
                        "Pass loss_target explicitly when return_loss=True."
                    )
                loss = self.mse_criterion(next_frames, loss_target)
            return next_frames, loss

        return next_frames
