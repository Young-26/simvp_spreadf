import unittest
from argparse import Namespace

import torch

import train as train_lib
from simvp.mim_model import MIM_Model
from simvp.wrapper import SimVPForecast


class MIMIntegrationTests(unittest.TestCase):
    def test_shape_smoke_simvp_forecast(self):
        model = SimVPForecast(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="mim",
            mim_hidden="8,8,8,8",
        )
        x = torch.randn(2, 8, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 64, 64))

    def test_openstl_training_smoke_returns_loss_and_backward(self):
        model = MIM_Model(
            shape_in=(8, 1, 64, 64),
            out_T=2,
            num_hidden="8,8,8,8",
            patch_size=4,
        )
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
        loss.backward()
        self.assertIsNotNone(model.conv_last.weight.grad)

    def test_openstl_reverse_schedule_smoke(self):
        model = MIM_Model(
            shape_in=(8, 1, 64, 64),
            out_T=2,
            num_hidden="8,8,8,8",
            patch_size=4,
            reverse_scheduled_sampling=True,
        )
        full_sequence = torch.randn(2, 10, 1, 64, 64)
        mask_true = torch.zeros(2, 8, 16, 16, 16)
        next_frames, loss = model(
            full_sequence,
            mask_true=mask_true,
            return_loss=True,
        )
        self.assertEqual(tuple(next_frames.shape), (2, 9, 1, 64, 64))
        self.assertEqual(loss.ndim, 0)

    def test_rss_switch_controls_model_and_mask_builder_consistently(self):
        for reverse_scheduled_sampling, expected_mask_steps in ((False, 1), (True, 8)):
            model = SimVPForecast(
                in_T=8,
                out_T=2,
                C=1,
                H=64,
                W=64,
                arch="mim",
                mim_hidden="8,8,8,8",
                reverse_scheduled_sampling=reverse_scheduled_sampling,
            )
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

    def test_internal_mim_block_state_is_reset_between_forward_calls(self):
        model = MIM_Model(
            shape_in=(8, 1, 64, 64),
            out_T=2,
            num_hidden="8,8,8,8",
            patch_size=4,
        )
        first_batch = torch.randn(2, 8, 1, 64, 64)
        second_batch = torch.randn(1, 8, 1, 64, 64)

        _ = model(first_batch)
        y = model(second_batch)
        self.assertEqual(tuple(y.shape), (1, 2, 1, 64, 64))

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
