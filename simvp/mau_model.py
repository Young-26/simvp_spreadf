import math
from collections import deque

import torch
import torch.nn as nn

from .convlstm_model import _parse_num_hidden, reshape_patch, reshape_patch_back


def _make_conv_norm_block(
    in_channels,
    out_channels,
    *,
    height,
    width,
    kernel_size,
    stride,
    padding,
    layer_norm,
    conv_bias,
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        # OpenSTL / official MAU keeps Conv2d bias even when LayerNorm follows it.
        # Keep that as the default here so benchmark ports can match the upstream cell math.
        bias=conv_bias,
    )
    if not layer_norm:
        return conv
    return nn.Sequential(conv, nn.LayerNorm([out_channels, height, width]))


class MAUCell(nn.Module):
    def __init__(
        self,
        in_channel,
        num_hidden,
        height,
        width,
        filter_size,
        stride,
        tau,
        cell_mode,
        layer_norm,
        conv_bias,
    ):
        super().__init__()

        self.num_hidden = int(num_hidden)
        self.padding = int(filter_size) // 2
        self.tau = int(tau)
        self.d = self.num_hidden * int(height) * int(width)
        self.cell_mode = str(cell_mode).strip().lower()
        if self.cell_mode not in {"residual", "normal"}:
            raise ValueError(
                f"Unsupported MAU cell_mode '{cell_mode}'. Available choices: ('residual', 'normal')."
            )

        self.conv_t = _make_conv_norm_block(
            in_channel,
            3 * self.num_hidden,
            height=height,
            width=width,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding,
            layer_norm=layer_norm,
            conv_bias=conv_bias,
        )
        self.conv_t_next = _make_conv_norm_block(
            in_channel,
            self.num_hidden,
            height=height,
            width=width,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding,
            layer_norm=layer_norm,
            conv_bias=conv_bias,
        )
        self.conv_s = _make_conv_norm_block(
            self.num_hidden,
            3 * self.num_hidden,
            height=height,
            width=width,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding,
            layer_norm=layer_norm,
            conv_bias=conv_bias,
        )
        self.conv_s_next = _make_conv_norm_block(
            self.num_hidden,
            self.num_hidden,
            height=height,
            width=width,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding,
            layer_norm=layer_norm,
            conv_bias=conv_bias,
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights = []
        scale = math.sqrt(float(self.d))
        for idx in range(self.tau):
            score = (s_att[idx] * s_next).sum(dim=(1, 2, 3)) / scale
            weights.append(score)
        weights = torch.stack(weights, dim=0).view(self.tau, S_t.shape[0], 1, 1, 1)
        weights = self.softmax(weights)

        T_trend = (t_att * weights).sum(dim=0)
        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1.0 - t_att_gate) * T_trend

        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1.0 - T_gate) * s_t
        S_new = S_gate * s_s + (1.0 - S_gate) * t_s

        if self.cell_mode == "residual":
            S_new = S_new + S_t
        return T_new, S_new


class MAU_Model(nn.Module):
    r"""MAU backbone adapted from OpenSTL for direct multi-step forecasting."""

    def __init__(
        self,
        shape_in,
        out_T,
        num_hidden="64,64,64,64",
        filter_size=5,
        patch_size=1,
        stride=1,
        sr_size=4,
        tau=5,
        cell_mode="normal",
        model_mode="normal",
        layer_norm=True,
        loss_mode="future_only",
        conv_bias=True,
    ):
        super().__init__()
        in_T, channels, height, width = shape_in

        self.in_T = int(in_T)
        self.out_T = int(out_T)
        self.input_channels = int(channels)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.sr_size = int(sr_size)
        self.tau = int(tau)
        self.cell_mode = str(cell_mode).strip().lower()
        self.model_mode = str(model_mode).strip().lower()
        self.layer_norm = bool(layer_norm)
        # Default to future_only in simvp_spreadf so MAU is benchmarked with the same
        # forecasting target convention used by most other models in this repo. Set
        # loss_mode='openstl_full' to reproduce OpenSTL's rollout-supervision objective.
        self.loss_mode = str(loss_mode).strip().lower()
        self.conv_bias = bool(conv_bias)
        self.num_hidden = _parse_num_hidden(num_hidden)
        self.num_layers = len(self.num_hidden)
        self.frame_channel = self.patch_size * self.patch_size * self.input_channels
        self.mse_criterion = nn.MSELoss()

        if self.num_layers < 1:
            raise ValueError("MAU requires at least one recurrent layer.")
        if len(set(self.num_hidden)) != 1:
            raise ValueError(
                "MAU requires identical hidden sizes across all layers because the OpenSTL-style "
                "temporal-state buffers and attention stacks assume one common channel width. "
                f"Received num_hidden={self.num_hidden}."
            )
        if self.stride != 1:
            raise ValueError(
                "simvp_spreadf MAU currently only supports mau_stride=1. "
                "The migrated OpenSTL cell stack assumes same-resolution recurrent states."
            )
        if self.sr_size < 1:
            raise ValueError(f"mau_sr_size must be >= 1, but got {self.sr_size}.")
        if self.sr_size & (self.sr_size - 1):
            raise ValueError(
                f"mau_sr_size must be a power of two because the encoder/decoder depth is log2(sr_size). Got {self.sr_size}."
            )
        if self.tau < 1:
            raise ValueError(f"mau_tau must be >= 1, but got {self.tau}.")
        if self.model_mode not in {"recall", "normal"}:
            raise ValueError(
                f"Unsupported MAU model_mode '{model_mode}'. Available choices: ('recall', 'normal')."
            )
        if self.loss_mode not in {"openstl_full", "future_only"}:
            raise ValueError(
                f"Unsupported MAU loss_mode '{loss_mode}'. Available choices: ('openstl_full', 'future_only')."
            )

        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        if patch_h * self.patch_size != height or patch_w * self.patch_size != width:
            raise ValueError(f"Input size {(height, width)} must be divisible by patch_size={self.patch_size}.")
        if patch_h % self.sr_size != 0 or patch_w % self.sr_size != 0:
            raise ValueError(
                f"Patch-space size {(patch_h, patch_w)} must be divisible by mau_sr_size={self.sr_size}."
            )

        state_h = patch_h // self.sr_size
        state_w = patch_w // self.sr_size
        downsample_stages = int(math.log2(self.sr_size))

        cell_list = []
        for idx in range(self.num_layers):
            in_channel = self.num_hidden[idx - 1] if idx > 0 else self.num_hidden[0]
            cell_list.append(
                MAUCell(
                    in_channel=in_channel,
                    num_hidden=self.num_hidden[idx],
                    height=state_h,
                    width=state_w,
                    filter_size=filter_size,
                    stride=stride,
                        tau=self.tau,
                        cell_mode=self.cell_mode,
                        layer_norm=self.layer_norm,
                        conv_bias=self.conv_bias,
                    )
                )
            self.cell_list = nn.ModuleList(cell_list)

        encoders = []
        encoder = nn.Sequential(
            nn.Conv2d(self.frame_channel, self.num_hidden[0], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        encoders.append(encoder)
        for _ in range(downsample_stages):
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.num_hidden[0],
                        self.num_hidden[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        for _ in range(max(downsample_stages - 1, 0)):
            decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.num_hidden[-1],
                        self.num_hidden[-1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
        if downsample_stages > 0:
            decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.num_hidden[-1],
                        self.num_hidden[-1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
            )
        self.decoders = nn.ModuleList(decoders)
        self.srcnn = nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)

    def _new_history_buffer(self, state_template):
        return deque((state_template.clone() for _ in range(self.tau)), maxlen=self.tau)

    def _mask_to_channel_first(self, mask_true, expected_steps, patch_h, patch_w):
        if expected_steps == 0:
            if mask_true is None:
                return None
            if mask_true.shape[1] != 0:
                raise ValueError(
                    f"MAU mask_true expects time length 0 for out_T={self.out_T}, but got {mask_true.shape[1]}."
                )
            return mask_true.new_zeros(mask_true.shape[0], 0, self.frame_channel, patch_h, patch_w)

        if mask_true is None:
            return None
        if mask_true.ndim != 5:
            raise ValueError(f"MAU mask_true expects 5 dims, but got {mask_true.shape}.")
        if mask_true.shape[1] != expected_steps:
            raise ValueError(f"MAU mask_true expects time length {expected_steps}, but got {mask_true.shape[1]}.")

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
            "MAU mask_true must be channel-first [B, T, Cp, Hp, Wp] or channel-last [B, T, Hp, Wp, Cp]. "
            f"Got {mask_true.shape} with expected patch grid {(patch_h, patch_w)} and channels {self.frame_channel}."
        )

    def _resolve_loss_pair(self, next_frames, frames_tensor, loss_target):
        if loss_target is None:
            if frames_tensor.shape[1] == self.in_T + self.out_T:
                if self.loss_mode == "openstl_full":
                    return next_frames, frames_tensor[:, 1:]
                return next_frames[:, -self.out_T :], frames_tensor[:, -self.out_T :]
            return None, None

        if loss_target.ndim != 5:
            raise ValueError(f"MAU loss_target expects [B, T, C, H, W], but got {loss_target.shape}.")
        if loss_target.shape[0] != next_frames.shape[0] or loss_target.shape[2:] != next_frames.shape[2:]:
            raise ValueError(f"MAU loss_target shape {loss_target.shape} is incompatible with predictions {next_frames.shape}.")
        if loss_target.shape[1] == next_frames.shape[1]:
            if self.loss_mode == "openstl_full":
                return next_frames, loss_target
            return next_frames[:, -self.out_T :], loss_target[:, -self.out_T :]
        if loss_target.shape[1] == self.out_T:
            return next_frames[:, -self.out_T :], loss_target
        raise ValueError(
            f"MAU loss_target time length must be {next_frames.shape[1]} or {self.out_T}, but got {loss_target.shape[1]}."
        )

    def forward(
        self,
        frames_tensor,
        mask_true=None,
        return_loss=False,
        loss_target=None,
    ):
        if frames_tensor.ndim != 5:
            raise ValueError(f"MAU expects [B, T, C, H, W], but got {frames_tensor.shape}.")

        batch, in_steps, channels, _, _ = frames_tensor.shape
        if channels != self.input_channels:
            raise ValueError(f"MAU was built for {self.input_channels} input channels, but got {channels}.")

        full_sequence_length = self.in_T + self.out_T
        if in_steps not in (self.in_T, full_sequence_length):
            raise ValueError(
                "MAU expects either the observed sequence length "
                f"{self.in_T} or the full training sequence length {full_sequence_length}, but got {in_steps}."
            )

        frames = reshape_patch(frames_tensor, self.patch_size)
        patch_h, patch_w = frames.shape[-2:]
        expected_mask_steps = max(self.out_T - 1, 0)
        mask_true = self._mask_to_channel_first(mask_true, expected_mask_steps, patch_h, patch_w)
        if mask_true is None:
            mask_true = frames.new_zeros(batch, expected_mask_steps, self.frame_channel, patch_h, patch_w)

        total_steps = full_sequence_length - 1 if in_steps == full_sequence_length else self.in_T + self.out_T - 1
        collect_all_steps = in_steps == full_sequence_length
        state_h = patch_h // self.sr_size
        state_w = patch_w // self.sr_size

        temporal_states = []
        temporal_buffers = []
        spatial_buffers = []
        for layer_idx in range(self.num_layers):
            zeros = frames.new_zeros(batch, self.num_hidden[layer_idx], state_h, state_w)
            temporal_states.append(zeros)
            history_channels = self.num_hidden[layer_idx - 1] if layer_idx > 0 else self.num_hidden[0]
            history_zeros = frames.new_zeros(batch, history_channels, state_h, state_w)
            temporal_buffers.append(self._new_history_buffer(history_zeros))
            spatial_buffers.append(self._new_history_buffer(history_zeros))

        x_gen = None
        next_frames = []

        for t in range(total_steps):
            if t < self.in_T:
                net = frames[:, t]
            elif t < in_steps:
                mask_idx = t - self.in_T
                net = mask_true[:, mask_idx] * frames[:, t] + (1.0 - mask_true[:, mask_idx]) * x_gen
            else:
                net = x_gen

            frame_feature = net
            encoded_features = []
            for encoder in self.encoders:
                frame_feature = encoder(frame_feature)
                encoded_features.append(frame_feature)

            spatial_state = frame_feature
            for layer_idx in range(self.num_layers):
                t_att = torch.stack(tuple(temporal_buffers[layer_idx]), dim=0)
                s_att = torch.stack(tuple(spatial_buffers[layer_idx]), dim=0)
                spatial_buffers[layer_idx].append(spatial_state)
                temporal_states[layer_idx], spatial_state = self.cell_list[layer_idx](
                    temporal_states[layer_idx],
                    spatial_state,
                    t_att,
                    s_att,
                )
                temporal_buffers[layer_idx].append(temporal_states[layer_idx])

            out = spatial_state
            for decoder_idx, decoder in enumerate(self.decoders):
                out = decoder(out)
                if self.model_mode == "recall":
                    out = out + encoded_features[-2 - decoder_idx]

            x_gen = self.srcnn(out)
            if collect_all_steps or t >= self.in_T - 1:
                next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        next_frames = reshape_patch_back(next_frames, self.patch_size, self.input_channels)

        if return_loss:
            loss_pred, loss_target = self._resolve_loss_pair(next_frames, frames_tensor, loss_target)
            if loss_pred is None or loss_target is None:
                raise ValueError(
                    "MAU needs either a full [x, y] sequence input or an explicit loss_target when return_loss=True."
                )
            loss = self.mse_criterion(loss_pred, loss_target)
            return next_frames, loss

        return next_frames
