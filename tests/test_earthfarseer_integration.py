import io
import unittest

import torch

import infer as infer_lib
import predict_all_preds as predict_lib
from simvp.wrapper import SimVPForecast


class EarthFarseerIntegrationTests(unittest.TestCase):
    def _build_model(self, **overrides):
        kwargs = dict(
            in_T=4,
            out_T=6,
            C=1,
            H=64,
            W=64,
            arch="earthfarseer",
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=4,
            earthfarseer_incep_ker="3,5",
            earthfarseer_groups=4,
            earthfarseer_num_interactions=1,
            earthfarseer_patch_size=16,
            earthfarseer_embed_dim=64,
            earthfarseer_depth=2,
            earthfarseer_mlp_ratio=2.0,
        )
        kwargs.update(overrides)
        return SimVPForecast(**kwargs)

    def test_shape_and_backward_smoke(self):
        model = self._build_model()
        x = torch.randn(1, 4, 1, 64, 64, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 6, 1, 64, 64))
        loss = y.square().mean()
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))

    def test_skip_branch_tracks_main_hyperparams(self):
        model = self._build_model(out_T=4, hid_S=10, hid_T=24, N_S=2, N_T=3)
        skip = model.backbone.skip_connection
        self.assertEqual(skip.enc.enc[0].conv.conv.out_channels, 10)
        self.assertEqual(len(skip.enc.enc), 2)
        self.assertEqual(skip.hid.N_T, 3)
        self.assertEqual(len(skip.dec.dec), 2)

    def test_specific_depth_kwargs_override_legacy_depth(self):
        model = self._build_model(
            out_T=4,
            earthfarseer_depth=5,
            earthfarseer_spatial_depth=1,
            earthfarseer_temporal_depth=3,
        )
        self.assertEqual(len(model.backbone.fotf_encoder.global_branch.blocks), 1)
        self.assertEqual(len(model.backbone.temporal_block.blocks), 3)

    def test_checkpoint_round_trip_rebuilds_earthfarseer(self):
        model = self._build_model(
            earthfarseer_spatial_depth=2,
            earthfarseer_temporal_depth=3,
        )
        ckpt_args = {
            "arch": "earthfarseer",
            "in_T": 4,
            "out_T": 6,
            "image_mode": "L",
            "image_size": 64,
            "hid_S": 8,
            "hid_T": 32,
            "N_S": 4,
            "N_T": 4,
            "earthfarseer_incep_ker": "3,5",
            "earthfarseer_groups": 4,
            "earthfarseer_num_interactions": 1,
            "earthfarseer_patch_size": 16,
            "earthfarseer_embed_dim": 64,
            "earthfarseer_depth": 2,
            "earthfarseer_spatial_depth": 2,
            "earthfarseer_temporal_depth": 3,
            "earthfarseer_mlp_ratio": 2.0,
            "earthfarseer_drop": 0.0,
            "earthfarseer_drop_path": 0.0,
        }

        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "args": ckpt_args}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)

        infer_model, infer_cfg = infer_lib.build_model_from_saved_args(
            loaded["args"],
            image_mode="L",
            image_size=64,
        )
        predict_model, predict_cfg = predict_lib.build_model_from_ckpt_args(
            loaded["args"],
            image_mode="L",
            image_size=64,
        )

        self.assertEqual(infer_cfg["arch"], "earthfarseer")
        self.assertEqual(predict_cfg["arch"], "earthfarseer")
        self.assertEqual(infer_cfg["earthfarseer_spatial_depth"], 2)
        self.assertEqual(predict_cfg["earthfarseer_temporal_depth"], 3)

        infer_model.load_state_dict(loaded["model"], strict=True)
        predict_model.load_state_dict(loaded["model"], strict=True)

        x = torch.randn(1, 4, 1, 64, 64)
        self.assertEqual(tuple(infer_model(x).shape), (1, 6, 1, 64, 64))
        self.assertEqual(tuple(predict_model(x).shape), (1, 6, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
