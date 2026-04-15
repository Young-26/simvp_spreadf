import unittest
from argparse import Namespace

import torch
from torch import nn

import train as train_lib
from simvp.predrnnpp_model import GHU, PredRNNpp_Model
from simvp.wrapper import SimVPForecast


class PredRNNppRecipeTests(unittest.TestCase):
    def test_ghu_smoke_without_layer_norm(self):
        ghu = GHU(
            in_channel=4,
            num_hidden=4,
            height=8,
            width=8,
            filter_size=3,
            stride=1,
            layer_norm=False,
        )
        x = torch.randn(2, 4, 8, 8)
        y = ghu(x, None)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_ghu_smoke_with_layer_norm(self):
        ghu = GHU(
            in_channel=4,
            num_hidden=4,
            height=8,
            width=8,
            filter_size=3,
            stride=1,
            layer_norm=True,
        )
        x = torch.randn(2, 4, 8, 8)
        z = torch.randn(2, 4, 8, 8)
        y = ghu(x, z)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(tuple(ghu.z_concat[1].normalized_shape), (8, 8, 8))
        self.assertEqual(tuple(ghu.x_concat[1].normalized_shape), (8, 8, 8))

    def test_shape_smoke_simvp_recipe(self):
        model = SimVPForecast(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="predrnnpp",
            predrnnpp_recipe="simvp",
            predrnnpp_hidden="8,8,8,8",
        )
        x = torch.randn(2, 8, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 64, 64))

    def test_shape_smoke_openstl_recipe(self):
        model = SimVPForecast(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="predrnnpp",
            predrnnpp_recipe="openstl",
            predrnnpp_hidden="8,8,8,8",
        )
        x = torch.randn(2, 8, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 64, 64))

    def test_openstl_training_smoke_returns_loss_and_backward(self):
        model = PredRNNpp_Model(
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
            recipe="openstl",
        )
        self.assertEqual(tuple(next_frames.shape), (2, 9, 1, 64, 64))
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertIsNotNone(model.conv_last.weight.grad)

    def test_openstl_reverse_schedule_smoke(self):
        model = PredRNNpp_Model(
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
            recipe="openstl",
        )
        self.assertEqual(tuple(next_frames.shape), (2, 9, 1, 64, 64))
        self.assertEqual(loss.ndim, 0)

    def test_scheduler_smoke_onecycle(self):
        args = Namespace(
            arch="predrnnpp",
            predrnnpp_recipe="openstl",
            opt="auto",
            sched="auto",
            warmup_epoch=0,
            epochs=2,
            lr=1e-3,
            weight_decay=1e-4,
        )
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        model = nn.Linear(4, 4)
        optimizer = train_lib.build_optimizer(args, model)
        scheduler, scheduler_step_mode = train_lib.build_lr_scheduler(args, optimizer, steps_per_epoch=3)

        self.assertEqual(type(optimizer).__name__, "Adam")
        self.assertEqual(type(scheduler).__name__, "OneCycleLR")
        self.assertEqual(scheduler_step_mode, "iter")

        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(2, 4)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

    def test_real_input_flag_shapes(self):
        args = Namespace(
            in_T=8,
            out_T=2,
            predrnnpp_patch_size=4,
            reverse_scheduled_sampling=False,
            scheduled_sampling=True,
            sampling_start_value=1.0,
            sampling_stop_iter=50000,
            r_sampling_step_1=25000,
            r_sampling_step_2=50000,
            r_exp_alpha=5000.0,
        )
        eta, mask = train_lib.build_predrnnpp_real_input_flag(
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
        self.assertEqual(tuple(mask.shape), (2, 1, 16, 16, 16))

        args.reverse_scheduled_sampling = True
        eta_reverse, reverse_mask = train_lib.build_predrnnpp_real_input_flag(
            args=args,
            batch_size=2,
            channels=1,
            height=64,
            width=64,
            device=torch.device("cpu"),
            eta=eta,
            itr=0,
        )
        self.assertAlmostEqual(eta_reverse, eta, places=6)
        self.assertEqual(tuple(reverse_mask.shape), (2, 8, 16, 16, 16))

    def test_patch_size_non_divisible_raises(self):
        with self.assertRaisesRegex(ValueError, "must be divisible by patch_size=4"):
            PredRNNpp_Model(
                shape_in=(8, 1, 62, 64),
                out_T=2,
                num_hidden="8,8,8,8",
                patch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
