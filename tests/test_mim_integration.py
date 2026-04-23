import unittest
from argparse import Namespace

import torch

import train as train_lib
from simvp.mim_model import MIMBlock, MIM_Model
from simvp.wrapper import SimVPForecast


class MIMIntegrationTests(unittest.TestCase):
    def _make_forecast_model(self, **overrides):
        kwargs = dict(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="mim",
            mim_hidden="8,8,8,8",
        )
        kwargs.update(overrides)
        return SimVPForecast(**kwargs)

    def _make_backbone(self, **overrides):
        kwargs = dict(
            shape_in=(8, 1, 64, 64),
            out_T=2,
            num_hidden="8,8,8,8",
            patch_size=4,
        )
        kwargs.update(overrides)
        return MIM_Model(**kwargs)

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
        mask_true = torch.zeros(2, 1, 16, 16, 16)

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
        self.assertIsNotNone(model.conv_last.weight.grad)

    def test_rss_switch_controls_model_and_mask_builder_consistently(self):
        for reverse_scheduled_sampling, expected_mask_steps in ((False, 1), (True, 8)):
            model = self._make_forecast_model(reverse_scheduled_sampling=reverse_scheduled_sampling)
            args = Namespace(
                in_T=8,
                out_T=2,
                mim_patch_size=4,
                reverse_scheduled_sampling=reverse_scheduled_sampling,
                scheduled_sampling=True,
                sampling_start_value=1.0,
                sampling_stop_iter=50000,
                r_sampling_step_1=25000,
                r_sampling_step_2=50000,
                r_exp_alpha=5000.0,
            )

            _, mask = train_lib.build_mim_real_input_flag(
                args=args,
                batch_size=2,
                channels=1,
                height=64,
                width=64,
                device=torch.device("cpu"),
                eta=1.0,
                itr=0,
            )

            self.assertEqual(model.backbone.reverse_scheduled_sampling, reverse_scheduled_sampling)
            self.assertEqual(mask.shape[1], expected_mask_steps)

    def test_internal_state_does_not_pollute_next_independent_forward(self):
        torch.manual_seed(3456)
        model = self._make_backbone()
        ref_input = torch.randn(2, 8, 1, 64, 64)
        other_input = torch.randn(1, 8, 1, 64, 64)

        pred_first = model(ref_input)
        mim_blocks = [layer for layer in model.stlstm_layers if isinstance(layer, MIMBlock)]
        self.assertTrue(mim_blocks)
        self.assertTrue(any(block.convlstm_c is not None for block in mim_blocks))

        _ = model(other_input)
        pred_second = model(ref_input)

        self.assertTrue(torch.equal(pred_first, pred_second))

    def test_non_uniform_hidden_sizes_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "requires identical hidden sizes"):
            self._make_backbone(num_hidden="8,16,8")

    def test_non_uniform_hidden_sizes_are_rejected_via_public_wrapper(self):
        with self.assertRaisesRegex(ValueError, "requires identical hidden sizes"):
            self._make_forecast_model(mim_hidden="8,16,8")

    def test_mim_stride_other_than_one_raises(self):
        with self.assertRaisesRegex(ValueError, "only supports mim_stride=1"):
            self._make_backbone(stride=2)

    def test_public_wrapper_rejects_mim_stride_other_than_one(self):
        with self.assertRaisesRegex(ValueError, "only supports mim_stride=1"):
            self._make_forecast_model(mim_stride=2)

    def test_patch_size_non_divisible_raises(self):
        with self.assertRaisesRegex(ValueError, "must be divisible by patch_size=4"):
            MIM_Model(
                shape_in=(8, 1, 62, 64),
                out_T=2,
                num_hidden="8,8,8,8",
                patch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
