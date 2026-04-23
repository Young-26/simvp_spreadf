import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

import train as train_lib
from simvp.mau_model import MAU_Model
from simvp.wrapper import SimVPForecast


class MAUIntegrationTests(unittest.TestCase):
    def _make_forecast_model(self, **overrides):
        kwargs = dict(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="mau",
            mau_hidden="8,8,8,8",
            mau_sr_size=4,
            mau_tau=3,
        )
        kwargs.update(overrides)
        return SimVPForecast(**kwargs)

    def _make_backbone(self, **overrides):
        kwargs = dict(
            shape_in=(8, 1, 64, 64),
            out_T=2,
            num_hidden="8,8,8,8",
            patch_size=2,
            sr_size=4,
            tau=3,
        )
        kwargs.update(overrides)
        return MAU_Model(**kwargs)

    def test_forward_shape_and_values_are_stable_under_fixed_seed(self):
        torch.manual_seed(1234)
        model = self._make_forecast_model()
        x = torch.randn(2, 8, 1, 64, 64)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 2, 1, 64, 64))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_return_loss_is_scalar_finite_and_backward_safe(self):
        torch.manual_seed(2345)
        model = self._make_backbone()
        full_sequence = torch.randn(2, 10, 1, 64, 64)
        mask_true = torch.zeros(2, 1, 32, 32, 4)

        next_frames, loss = model(
            full_sequence,
            mask_true=mask_true,
            return_loss=True,
        )

        self.assertEqual(tuple(next_frames.shape), (2, 9, 1, 64, 64))
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.srcnn.weight.grad)

    def test_loss_mode_switches_supervision_protocol(self):
        torch.manual_seed(3456)
        model_full = self._make_backbone(loss_mode="openstl_full")
        model_future = self._make_backbone(loss_mode="future_only")
        model_future.load_state_dict(model_full.state_dict(), strict=True)

        full_sequence = torch.randn(2, 10, 1, 64, 64)
        mask_true = torch.zeros(2, 1, 32, 32, 4)

        pred_full, loss_full = model_full(full_sequence, mask_true=mask_true, return_loss=True)
        pred_future, loss_future = model_future(full_sequence, mask_true=mask_true, return_loss=True)

        manual_full = F.mse_loss(pred_full, full_sequence[:, 1:])
        manual_future = F.mse_loss(pred_future[:, -2:], full_sequence[:, -2:])

        torch.testing.assert_close(pred_full, pred_future)
        torch.testing.assert_close(loss_full, manual_full)
        torch.testing.assert_close(loss_future, manual_future)

    def test_real_input_flag_shapes_follow_patch_grid(self):
        args = Namespace(
            in_T=8,
            out_T=2,
            mau_patch_size=2,
            scheduled_sampling=True,
            sampling_start_value=1.0,
            sampling_stop_iter=50000,
            r_sampling_step_1=25000,
            r_sampling_step_2=50000,
            r_exp_alpha=5000.0,
        )
        eta, mask = train_lib.build_mau_real_input_flag(
            args=args,
            batch_size=2,
            channels=1,
            height=64,
            width=64,
            device=torch.device("cpu"),
            eta=1.0,
            itr=0,
        )

        self.assertAlmostEqual(eta, 1.0 - (1.0 / 50000.0), places=6)
        self.assertEqual(tuple(mask.shape), (2, 1, 32, 32, 4))

    def test_reverse_scheduled_sampling_is_rejected_via_public_wrapper(self):
        with self.assertRaisesRegex(ValueError, "does not support reverse_scheduled_sampling"):
            self._make_forecast_model(reverse_scheduled_sampling=True)

    def test_non_uniform_hidden_sizes_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "requires identical hidden sizes"):
            self._make_backbone(num_hidden="8,16,8")

    def test_non_power_of_two_sr_size_raises(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            self._make_backbone(sr_size=3)

    def test_history_buffer_uses_fixed_maxlen(self):
        model = self._make_backbone(tau=2)
        template = torch.zeros(1, 8, 8, 8)
        buffer = model._new_history_buffer(template)

        self.assertEqual(buffer.maxlen, 2)
        for value in range(5):
            buffer.append(torch.full_like(template, float(value)))
        self.assertEqual(len(buffer), 2)
        self.assertTrue(torch.equal(buffer[0], torch.full_like(template, 3.0)))
        self.assertTrue(torch.equal(buffer[1], torch.full_like(template, 4.0)))

    def test_default_conv_bias_matches_upstream_mau(self):
        model = self._make_backbone(layer_norm=True, conv_bias=True)
        self.assertIsNotNone(model.cell_list[0].conv_t[0].bias)


if __name__ == "__main__":
    unittest.main()
