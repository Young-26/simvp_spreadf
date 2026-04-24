import io
import unittest
from collections import OrderedDict

import torch

from simvp.model import MetaBlock, SimVP
from simvp.predformer_facts_model import PredFormerFacTS_Model
from simvp.predformer_quadruplet_tsst_model import PredFormerQuadrupletTSST_Model
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

    def test_shape_and_backward_smoke_moganet(self):
        self._run_forward_and_backward("moganet")

    def test_shape_and_backward_smoke_moga_alias(self):
        self._run_forward_and_backward("moga")

    def test_direct_simvp_constructor_accepts_moga_alias(self):
        model = SimVP(
            shape_in=(8, 1, 64, 64),
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=4,
            model_type="moga",
        )
        x = torch.randn(1, 8, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 8, 1, 64, 64))

    def test_metablock_accepts_moga_alias(self):
        block = MetaBlock(
            in_channels=32,
            out_channels=32,
            model_type="moga",
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
        )
        x = torch.randn(2, 32, 8, 8)
        y = block(x)
        self.assertEqual(tuple(y.shape), (2, 32, 8, 8))

    def test_moganet_state_dict_uses_openstl_key_names(self):
        block = MetaBlock(
            in_channels=32,
            out_channels=32,
            model_type="moga",
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
        )
        keys = set(block.state_dict().keys())
        self.assertIn("block.attn.value.DW_conv0.weight", keys)
        self.assertIn("block.attn.value.DW_conv1.weight", keys)
        self.assertIn("block.attn.value.DW_conv2.weight", keys)
        self.assertIn("block.attn.value.PW_conv.weight", keys)
        self.assertNotIn("block.attn.value.dw_conv0.weight", keys)
        self.assertNotIn("block.attn.value.pw_conv.weight", keys)

    def test_moganet_legacy_lowercase_state_dict_keys_still_load(self):
        block = MetaBlock(
            in_channels=32,
            out_channels=32,
            model_type="moga",
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
        )
        legacy_state = OrderedDict()
        for key, value in block.state_dict().items():
            legacy_key = (
                key.replace(".DW_conv0.", ".dw_conv0.")
                .replace(".DW_conv1.", ".dw_conv1.")
                .replace(".DW_conv2.", ".dw_conv2.")
                .replace(".PW_conv.", ".pw_conv.")
            )
            legacy_state[legacy_key] = value.clone()

        reloaded = MetaBlock(
            in_channels=32,
            out_channels=32,
            model_type="moga",
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
        )
        result = reloaded.load_state_dict(legacy_state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

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

    def test_predformer_facts_shape_and_backward_smoke(self):
        model = SimVPForecast(
            in_T=4,
            out_T=2,
            C=1,
            H=32,
            W=32,
            arch="predformer_facts",
            predformer_patch_size=8,
            predformer_dim=64,
            predformer_heads=4,
            predformer_dim_head=16,
            predformer_depth=2,
        )
        x = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 32, 32))
        y.square().mean().backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))

    def test_predformer_facts_supports_odd_predformer_dim(self):
        model = SimVPForecast(
            in_T=4,
            out_T=2,
            C=1,
            H=32,
            W=32,
            arch="predformer_facts",
            predformer_patch_size=8,
            predformer_dim=65,
            predformer_heads=5,
            predformer_dim_head=13,
            predformer_depth=1,
        )
        x = torch.randn(1, 4, 1, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 2, 1, 32, 32))

    def test_predformer_facts_autoregressive_rollout_supports_out_t_gt_in_t(self):
        model = SimVPForecast(
            in_T=4,
            out_T=6,
            C=1,
            H=32,
            W=32,
            arch="predformer_facts",
            predformer_patch_size=8,
            predformer_dim=64,
            predformer_heads=4,
            predformer_dim_head=16,
            predformer_depth=1,
        )
        x = torch.randn(1, 4, 1, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 6, 1, 32, 32))

    def test_predformer_facts_position_embedding_is_buffer_not_trainable_parameter(self):
        model = PredFormerFacTS_Model(
            shape_in=(4, 1, 32, 32),
            patch_size=8,
            dim=65,
            heads=5,
            dim_head=13,
            depth=1,
        )
        self.assertIn("pos_embedding", model.state_dict())
        self.assertIn("pos_embedding", dict(model.named_buffers()))
        self.assertNotIn("pos_embedding", dict(model.named_parameters()))

    def test_predformer_facts_state_dict_round_trip_preserves_buffer_and_forward(self):
        model = PredFormerFacTS_Model(
            shape_in=(4, 1, 32, 32),
            patch_size=8,
            dim=65,
            heads=5,
            dim_head=13,
            depth=1,
        )
        model.eval()
        x = torch.randn(1, 4, 1, 32, 32)
        y = model(x)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        reloaded_state = torch.load(buffer, map_location="cpu", weights_only=False)

        reloaded = PredFormerFacTS_Model(
            shape_in=(4, 1, 32, 32),
            patch_size=8,
            dim=65,
            heads=5,
            dim_head=13,
            depth=1,
        )
        load_result = reloaded.load_state_dict(reloaded_state, strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertIn("pos_embedding", dict(reloaded.named_buffers()))

        reloaded.eval()
        y_reloaded = reloaded(x)
        torch.testing.assert_close(y, y_reloaded)

    def test_predformer_quadruplet_tsst_shape_and_backward_smoke(self):
        model = SimVPForecast(
            in_T=4,
            out_T=2,
            C=1,
            H=32,
            W=32,
            arch="predformer_quadruplet_tsst",
            predformer_patch_size=8,
            predformer_dim=64,
            predformer_heads=4,
            predformer_dim_head=16,
            predformer_depth=2,
        )
        x = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 1, 32, 32))
        y.square().mean().backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))

    def test_predformer_quadruplet_tsst_supports_odd_predformer_dim(self):
        model = SimVPForecast(
            in_T=4,
            out_T=2,
            C=1,
            H=32,
            W=32,
            arch="predformer_quadruplet_tsst",
            predformer_patch_size=8,
            predformer_dim=65,
            predformer_heads=5,
            predformer_dim_head=13,
            predformer_depth=1,
        )
        x = torch.randn(1, 4, 1, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 2, 1, 32, 32))

    def test_predformer_quadruplet_tsst_autoregressive_rollout_supports_out_t_gt_in_t(self):
        model = SimVPForecast(
            in_T=4,
            out_T=6,
            C=1,
            H=32,
            W=32,
            arch="predformer_quadruplet_tsst",
            predformer_patch_size=8,
            predformer_dim=64,
            predformer_heads=4,
            predformer_dim_head=16,
            predformer_depth=1,
        )
        x = torch.randn(1, 4, 1, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 6, 1, 32, 32))

    def test_predformer_quadruplet_tsst_position_embedding_is_buffer_not_trainable_parameter(self):
        model = PredFormerQuadrupletTSST_Model(
            shape_in=(4, 1, 32, 32),
            patch_size=8,
            dim=65,
            heads=5,
            dim_head=13,
            depth=1,
        )
        self.assertIn("pos_embedding", model.state_dict())
        self.assertIn("pos_embedding", dict(model.named_buffers()))
        self.assertNotIn("pos_embedding", dict(model.named_parameters()))

    def test_predformer_quadruplet_tsst_state_dict_round_trip_preserves_buffer_and_forward(self):
        model = PredFormerQuadrupletTSST_Model(
            shape_in=(4, 1, 32, 32),
            patch_size=8,
            dim=65,
            heads=5,
            dim_head=13,
            depth=1,
        )
        model.eval()
        x = torch.randn(1, 4, 1, 32, 32)
        y = model(x)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        reloaded_state = torch.load(buffer, map_location="cpu", weights_only=False)

        reloaded = PredFormerQuadrupletTSST_Model(
            shape_in=(4, 1, 32, 32),
            patch_size=8,
            dim=65,
            heads=5,
            dim_head=13,
            depth=1,
        )
        load_result = reloaded.load_state_dict(reloaded_state, strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertIn("pos_embedding", dict(reloaded.named_buffers()))

        reloaded.eval()
        y_reloaded = reloaded(x)
        torch.testing.assert_close(y, y_reloaded)


if __name__ == "__main__":
    unittest.main()
