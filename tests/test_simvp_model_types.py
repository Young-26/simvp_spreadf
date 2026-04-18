import unittest

import torch

from simvp.wrapper import SimVPForecast


class SimVPModelTypeTests(unittest.TestCase):
    def _run_forward_and_backward(self, model_type: str):
        model = SimVPForecast(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="simvp",
            simvp_model_type=model_type,
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=4,
        )
        x = torch.randn(2, 8, 1, 64, 64, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 64, 64))
        loss = y.square().mean()
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))

    def test_shape_smoke_incepu(self):
        self._run_forward_and_backward("incepu")

    def test_shape_and_backward_smoke_gsta(self):
        self._run_forward_and_backward("gsta")

    def test_shape_and_backward_smoke_v2_alias(self):
        self._run_forward_and_backward("v2")

    def test_shape_and_backward_smoke_simvpv2_alias(self):
        self._run_forward_and_backward("simvpv2")

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            SimVPForecast(
                in_T=8,
                out_T=2,
                C=1,
                H=64,
                W=64,
                arch="simvp",
                simvp_model_type="invalid",
            )


if __name__ == "__main__":
    unittest.main()
