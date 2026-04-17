import unittest
from argparse import Namespace

import torch
from torch import nn

import train as train_lib
from simvp.tau_model import tau_diff_div_reg
from simvp.wrapper import SimVPForecast


class TauIntegrationTests(unittest.TestCase):
    def test_shape_smoke(self):
        model = SimVPForecast(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="tau",
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=4,
        )
        x = torch.randn(2, 8, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 64, 64))

    def test_diff_div_reg_backward_and_short_seq_zero(self):
        pred = torch.randn(2, 4, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 1, 16, 16)
        loss = tau_diff_div_reg(pred, target)
        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)

        short_loss = tau_diff_div_reg(
            torch.randn(2, 2, 1, 16, 16),
            torch.randn(2, 2, 1, 16, 16),
        )
        self.assertEqual(short_loss.ndim, 0)
        self.assertEqual(short_loss.item(), 0.0)

    def test_scheduler_smoke_cosine(self):
        args = Namespace(
            arch="tau",
            predrnnpp_recipe="simvp",
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
        self.assertEqual(type(scheduler).__name__, "CosineAnnealingLR")
        self.assertEqual(scheduler_step_mode, "epoch")

        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(2, 4)).sum()
            loss.backward()
            optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    unittest.main()
