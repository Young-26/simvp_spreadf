import unittest
from argparse import Namespace

import torch
from torch import nn

import train as train_lib
from simvp.predrnnpp_model import GHU, PredRNNpp_Model
from simvp.wrapper import SimVPForecast


class _StepCounterOptimizer:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class _StepCounterScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class _FakeScaler:
    def __init__(self, old_scale: float, new_scale: float, execute_optimizer_step: bool):
        self._scale = float(old_scale)
        self._new_scale = float(new_scale)
        self.execute_optimizer_step = bool(execute_optimizer_step)
        self.step_calls = 0
        self.update_calls = 0

    def step(self, optimizer):
        self.step_calls += 1
        if self.execute_optimizer_step:
            optimizer.step()

    def update(self):
        self.update_calls += 1
        self._scale = self._new_scale

    def get_scale(self):
        return self._scale


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

    def test_iter_scheduler_steps_after_non_amp_optimizer_step(self):
        optimizer = _StepCounterOptimizer()
        scheduler = _StepCounterScheduler()
        scaler = _FakeScaler(old_scale=1.0, new_scale=1.0, execute_optimizer_step=True)

        optimizer_step_executed = train_lib.step_optimizer_and_maybe_step_scheduler(
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_mode="iter",
            amp_enabled=False,
        )

        self.assertTrue(optimizer_step_executed)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(scheduler.step_calls, 1)
        self.assertEqual(scaler.step_calls, 0)
        self.assertEqual(scaler.update_calls, 0)

    def test_iter_scheduler_does_not_advance_when_amp_skips_step(self):
        optimizer = _StepCounterOptimizer()
        scheduler = _StepCounterScheduler()
        scaler = _FakeScaler(old_scale=65536.0, new_scale=32768.0, execute_optimizer_step=False)

        optimizer_step_executed = train_lib.step_optimizer_and_maybe_step_scheduler(
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_mode="iter",
            amp_enabled=True,
        )

        self.assertFalse(optimizer_step_executed)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(scheduler.step_calls, 0)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)

    def test_iter_scheduler_advances_when_amp_step_is_valid(self):
        optimizer = _StepCounterOptimizer()
        scheduler = _StepCounterScheduler()
        scaler = _FakeScaler(old_scale=65536.0, new_scale=65536.0, execute_optimizer_step=True)

        optimizer_step_executed = train_lib.step_optimizer_and_maybe_step_scheduler(
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_mode="iter",
            amp_enabled=True,
        )

        self.assertTrue(optimizer_step_executed)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(scheduler.step_calls, 1)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)

    def test_epoch_scheduler_is_not_advanced_inside_train_step_helper(self):
        optimizer = _StepCounterOptimizer()
        scheduler = _StepCounterScheduler()
        scaler = _FakeScaler(old_scale=65536.0, new_scale=65536.0, execute_optimizer_step=True)

        optimizer_step_executed = train_lib.step_optimizer_and_maybe_step_scheduler(
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_mode="epoch",
            amp_enabled=True,
        )

        self.assertTrue(optimizer_step_executed)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(scheduler.step_calls, 0)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)

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
