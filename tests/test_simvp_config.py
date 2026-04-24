import contextlib
import io
import unittest
from argparse import Namespace

import torch

import infer as infer_lib
import predict_all_preds as predict_lib
import train as train_lib
from simvp.simvp_config import (
    SIMVP_MODEL_TYPE_ALIASES,
    SIMVP_MODEL_TYPE_CHOICES,
    SIMVP_OPENSTL_TRAIN_PRESET,
    SIMVP_RECIPE_CHOICES,
    build_forecast_model_kwargs_from_config,
    get_effective_simvp_recipe,
    is_simvp_openstl_recipe,
    normalize_simvp_model_type,
)
from simvp.wrapper import SUPPORTED_ARCHS, SimVPForecast


def _get_action_choices(parser, dest: str):
    for action in parser._actions:
        if action.dest == dest:
            return tuple(action.choices) if action.choices is not None else None
    raise KeyError(dest)


def _get_action_help(parser, dest: str):
    for action in parser._actions:
        if action.dest == dest:
            return action.help
    raise KeyError(dest)


class SimVPConfigTests(unittest.TestCase):
    def _make_train_args(self, **overrides):
        base = dict(
            arch="simvp",
            simvp_model_type="gsta",
            simvp_recipe="openstl",
            predrnnpp_recipe="simvp",
            predformer_loss="mse",
            predformer_recipe="auto",
            predformer_transformer_depth=1,
            opt="auto",
            sched="auto",
            warmup_epoch=3,
            hid_S=32,
            hid_T=128,
            N_S=4,
            N_T=4,
            lr=2e-4,
            weight_decay=1e-4,
            batch_size=4,
            simvp_drop_path=0.1,
            train_manifest="train.jsonl",
            val_manifest="val.jsonl",
        )
        base.update(overrides)
        return Namespace(**base)

    def test_simvp_model_type_alias_normalization(self):
        self.assertEqual(normalize_simvp_model_type(None), "incepu")
        self.assertEqual(normalize_simvp_model_type("incepu"), "incepu")
        self.assertEqual(normalize_simvp_model_type("gsta"), "gsta")
        self.assertEqual(normalize_simvp_model_type("moga"), "moganet")
        self.assertEqual(normalize_simvp_model_type("moganet"), "moganet")
        self.assertEqual(normalize_simvp_model_type("v1"), "incepu")
        self.assertEqual(normalize_simvp_model_type("simvpv1"), "incepu")
        self.assertEqual(normalize_simvp_model_type("v2"), "gsta")
        self.assertEqual(normalize_simvp_model_type("simvpv2"), "gsta")
        with self.assertRaises(ValueError):
            normalize_simvp_model_type("invalid")

    def test_cli_choices_are_consistent(self):
        self.assertEqual(_get_action_choices(train_lib.build_parser(), "arch"), SUPPORTED_ARCHS)
        self.assertEqual(_get_action_choices(infer_lib.build_parser(), "arch"), SUPPORTED_ARCHS)
        self.assertEqual(_get_action_choices(predict_lib.build_parser(), "arch"), SUPPORTED_ARCHS)
        self.assertEqual(_get_action_choices(train_lib.build_parser(), "simvp_model_type"), SIMVP_MODEL_TYPE_ALIASES)
        self.assertEqual(_get_action_choices(infer_lib.build_parser(), "simvp_model_type"), SIMVP_MODEL_TYPE_ALIASES)
        self.assertEqual(_get_action_choices(predict_lib.build_parser(), "simvp_model_type"), SIMVP_MODEL_TYPE_ALIASES)
        self.assertEqual(_get_action_choices(train_lib.build_parser(), "simvp_recipe"), SIMVP_RECIPE_CHOICES)
        self.assertEqual(_get_action_choices(infer_lib.build_parser(), "simvp_recipe"), SIMVP_RECIPE_CHOICES)
        self.assertEqual(_get_action_choices(predict_lib.build_parser(), "simvp_recipe"), SIMVP_RECIPE_CHOICES)
        self.assertEqual(_get_action_choices(train_lib.build_parser(), "predformer_loss"), ("mae", "mse", "hybrid"))
        self.assertEqual(
            _get_action_choices(train_lib.build_parser(), "predformer_recipe"),
            ("auto", "generic", "mmnist", "taxibj", "human", "weather"),
        )

    def test_train_parser_documents_mim_constraints(self):
        parser = train_lib.build_parser()
        mim_hidden_help = _get_action_help(parser, "mim_hidden")
        mim_stride_help = _get_action_help(parser, "mim_stride")

        self.assertIn("identical widths", mim_hidden_help)
        self.assertIn("only supports 1", mim_stride_help)

    def test_train_and_infer_parsers_accept_aliases(self):
        train_args = train_lib.parse_args(
            [
                "--train_manifest",
                "train.jsonl",
                "--val_manifest",
                "val.jsonl",
                "--simvp_model_type",
                "simvpv2",
                "--simvp_recipe",
                "openstl",
            ]
        )
        infer_args = infer_lib.parse_args(
            [
                "--manifest",
                "val.jsonl",
                "--checkpoint",
                "best.ckpt",
                "--simvp_model_type",
                "v2",
                "--simvp_recipe",
                "openstl",
            ]
        )
        self.assertEqual(train_args.simvp_model_type, "simvpv2")
        self.assertEqual(infer_args.simvp_model_type, "v2")

        moga_args = infer_lib.parse_args(
            [
                "--manifest",
                "val.jsonl",
                "--checkpoint",
                "best.ckpt",
                "--simvp_model_type",
                "moga",
            ]
        )
        self.assertEqual(moga_args.simvp_model_type, "moga")

    def test_infer_parser_rejects_invalid_model_type(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                infer_lib.parse_args(
                    [
                        "--manifest",
                        "val.jsonl",
                        "--checkpoint",
                        "best.ckpt",
                        "--simvp_model_type",
                        "invalid",
                    ]
                )

    def test_simvp_gsta_openstl_recipe_resolves_auto_optimizer_scheduler(self):
        args = self._make_train_args(simvp_model_type="v2", simvp_recipe="auto")
        args = train_lib.apply_simvp_recipe_defaults(args, explicit_cli_args=set())
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        self.assertEqual(args.simvp_model_type, "gsta")
        self.assertEqual(args.simvp_recipe, "auto")
        self.assertEqual(get_effective_simvp_recipe(args.arch, args.simvp_model_type, args.simvp_recipe), "openstl")
        self.assertEqual(args.opt, "adam")
        self.assertEqual(args.sched, "onecycle")
        self.assertEqual(args.hid_S, SIMVP_OPENSTL_TRAIN_PRESET["hid_S"])
        self.assertEqual(args.hid_T, SIMVP_OPENSTL_TRAIN_PRESET["hid_T"])
        self.assertEqual(args.N_S, SIMVP_OPENSTL_TRAIN_PRESET["N_S"])
        self.assertEqual(args.N_T, SIMVP_OPENSTL_TRAIN_PRESET["N_T"])
        self.assertEqual(args.lr, SIMVP_OPENSTL_TRAIN_PRESET["lr"])
        self.assertEqual(args.batch_size, SIMVP_OPENSTL_TRAIN_PRESET["batch_size"])
        self.assertEqual(args.simvp_drop_path, SIMVP_OPENSTL_TRAIN_PRESET["simvp_drop_path"])
        self.assertEqual(args.warmup_epoch, 0)

    def test_simvp_moganet_openstl_recipe_resolves_auto_optimizer_scheduler(self):
        args = self._make_train_args(simvp_model_type="moga", simvp_recipe="auto")
        args = train_lib.apply_simvp_recipe_defaults(args, explicit_cli_args=set())
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        self.assertEqual(args.simvp_model_type, "moganet")
        self.assertEqual(get_effective_simvp_recipe(args.arch, args.simvp_model_type, args.simvp_recipe), "openstl")
        self.assertTrue(is_simvp_openstl_recipe(args.arch, args.simvp_model_type, args.simvp_recipe))
        self.assertEqual(args.opt, "adam")
        self.assertEqual(args.sched, "onecycle")
        self.assertEqual(args.hid_S, SIMVP_OPENSTL_TRAIN_PRESET["hid_S"])
        self.assertEqual(args.hid_T, SIMVP_OPENSTL_TRAIN_PRESET["hid_T"])
        self.assertEqual(args.N_S, SIMVP_OPENSTL_TRAIN_PRESET["N_S"])
        self.assertEqual(args.N_T, SIMVP_OPENSTL_TRAIN_PRESET["N_T"])
        self.assertEqual(args.lr, SIMVP_OPENSTL_TRAIN_PRESET["lr"])
        self.assertEqual(args.batch_size, SIMVP_OPENSTL_TRAIN_PRESET["batch_size"])
        self.assertEqual(args.warmup_epoch, 0)

    def test_simvp_moganet_openstl_recipe_uses_mae_openstl_loss_mode(self):
        args = self._make_train_args(simvp_model_type="moga", simvp_recipe="openstl")
        args = train_lib.apply_simvp_recipe_defaults(args, explicit_cli_args=set())

        self.assertEqual(args.simvp_model_type, "moganet")
        self.assertTrue(train_lib.uses_simvp_moganet_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "mae_openstl")

    def test_simvp_gsta_openstl_recipe_keeps_local_reconstruction_loss_mode(self):
        args = self._make_train_args(simvp_model_type="gsta", simvp_recipe="openstl")
        args = train_lib.apply_simvp_recipe_defaults(args, explicit_cli_args=set())

        self.assertFalse(train_lib.uses_simvp_moganet_openstl_loss(args))
        self.assertTrue(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "lambda_global*global_l1 + lambda_local*local_l1")

    def test_earthfarseer_uses_mse_openstl_loss_mode(self):
        args = self._make_train_args(arch="earthfarseer", simvp_model_type="gsta", simvp_recipe="openstl")

        self.assertTrue(train_lib.uses_earthfarseer_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "mse_openstl")

    def test_mim_uses_openstl_optimizer_scheduler_and_loss_mode(self):
        args = self._make_train_args(arch="mim", mim_hidden="8,8,8,8")
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        self.assertTrue(train_lib.uses_mim_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(args.opt, "adam")
        self.assertEqual(args.sched, "onecycle")
        self.assertEqual(args.warmup_epoch, 0)
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "mse_openstl")

    def test_mim_requires_ddp_find_unused_parameters(self):
        args = self._make_train_args(arch="mim", mim_hidden="8,8,8,8")
        self.assertTrue(train_lib.should_enable_ddp_find_unused_parameters(args))

    def test_predformer_facts_uses_mse_openstl_loss_mode_by_default(self):
        args = self._make_train_args(arch="predformer_facts", simvp_model_type="gsta", simvp_recipe="openstl")

        self.assertTrue(train_lib.uses_predformer_openstl_loss(args))
        self.assertTrue(train_lib.uses_predformer_facts_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "mse_openstl")

    def test_predformer_quadruplet_tsst_uses_mse_openstl_loss_mode_by_default(self):
        args = self._make_train_args(
            arch="predformer_quadruplet_tsst",
            simvp_model_type="gsta",
            simvp_recipe="openstl",
        )

        self.assertTrue(train_lib.uses_predformer_openstl_loss(args))
        self.assertTrue(train_lib.uses_predformer_facts_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "mse_openstl")

    def test_predformer_facts_supports_mae_openstl_loss_mode(self):
        args = self._make_train_args(
            arch="predformer_facts",
            simvp_model_type="gsta",
            simvp_recipe="openstl",
            predformer_loss="mae",
        )

        self.assertTrue(train_lib.uses_predformer_openstl_loss(args))
        self.assertTrue(train_lib.uses_predformer_facts_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "mae_openstl")

    def test_predformer_facts_supports_hybrid_loss_mode(self):
        args = self._make_train_args(
            arch="predformer_facts",
            simvp_model_type="gsta",
            simvp_recipe="openstl",
            predformer_loss="hybrid",
        )

        self.assertTrue(train_lib.uses_predformer_facts_openstl_loss(args))
        self.assertFalse(train_lib.uses_local_reconstruction_loss(args))
        self.assertEqual(train_lib.resolve_train_loss_mode(args), "0.8000*MAE + 0.2000*MSE")

    def test_predformer_generic_recipe_resolves_auto_optimizer_scheduler_and_defaults(self):
        args = self._make_train_args(
            arch="predformer_quadruplet_tsst",
            predformer_recipe="generic",
            predformer_loss="mae",
            weight_decay=1e-4,
        )
        args = train_lib.apply_predformer_recipe_defaults(args, explicit_cli_args=set())
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        self.assertEqual(train_lib.get_predformer_recipe(args), "generic")
        self.assertEqual(args.opt, "adamw")
        self.assertEqual(args.sched, "onecycle")
        self.assertEqual(args.weight_decay, 1e-2)
        self.assertEqual(args.predformer_loss, "mse")
        self.assertEqual(args.warmup_epoch, 0)

    def test_predformer_human_recipe_resolves_auto_cosine_scheduler_and_defaults(self):
        args = self._make_train_args(
            arch="predformer_quadruplet_tsst",
            predformer_recipe="human",
            predformer_loss="mae",
            weight_decay=1e-4,
        )
        args = train_lib.apply_predformer_recipe_defaults(args, explicit_cli_args=set())
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        self.assertEqual(train_lib.get_predformer_recipe(args), "human")
        self.assertEqual(args.opt, "adamw")
        self.assertEqual(args.sched, "cosine")
        self.assertEqual(args.weight_decay, 1e-2)
        self.assertEqual(args.predformer_loss, "mse")
        self.assertEqual(args.warmup_epoch, 0)

    def test_predformer_recipe_auto_infers_dataset_from_manifest_paths(self):
        args = self._make_train_args(
            arch="predformer_quadruplet_tsst",
            predformer_recipe="auto",
            train_manifest="manifests/human/train.jsonl",
            val_manifest="manifests/human/val.jsonl",
        )
        self.assertEqual(train_lib.get_predformer_recipe(args), "human")

        args = train_lib.apply_predformer_recipe_defaults(args, explicit_cli_args=set())
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)
        self.assertEqual(args.sched, "cosine")

    def test_predformer_recipe_defaults_respect_explicit_overrides(self):
        args = self._make_train_args(
            arch="predformer_quadruplet_tsst",
            predformer_recipe="human",
            predformer_loss="hybrid",
            weight_decay=5e-4,
        )
        args = train_lib.apply_predformer_recipe_defaults(
            args,
            explicit_cli_args={"predformer_loss", "weight_decay"},
        )

        self.assertEqual(args.predformer_loss, "hybrid")
        self.assertEqual(args.weight_decay, 5e-4)

    def test_explicit_simvp_recipe_overrides_take_priority(self):
        args = self._make_train_args(
            simvp_model_type="simvpv2",
            simvp_recipe="openstl",
            hid_T=1024,
            lr=5e-4,
        )
        args = train_lib.apply_simvp_recipe_defaults(args, explicit_cli_args={"hid_T", "lr"})
        args = train_lib.resolve_optimizer_config(args)
        args = train_lib.resolve_scheduler_config(args)

        self.assertEqual(args.simvp_model_type, "gsta")
        self.assertEqual(args.hid_T, 1024)
        self.assertEqual(args.lr, 5e-4)
        self.assertEqual(args.hid_S, SIMVP_OPENSTL_TRAIN_PRESET["hid_S"])
        self.assertEqual(args.opt, "adam")
        self.assertEqual(args.sched, "onecycle")

    def test_checkpoint_round_trip_rebuilds_alias_to_gsta(self):
        model = SimVPForecast(
            in_T=8,
            out_T=2,
            C=1,
            H=64,
            W=64,
            arch="simvp",
            simvp_model_type="gsta",
            simvp_recipe="openstl",
            hid_S=8,
            hid_T=32,
            N_S=4,
            N_T=4,
        )
        ckpt_args = {
            "arch": "simvp",
            "in_T": 8,
            "out_T": 2,
            "image_mode": "L",
            "image_size": 64,
            "hid_S": 8,
            "hid_T": 32,
            "N_S": 4,
            "N_T": 4,
            "simvp_model_type": "simvpv2",
            "simvp_recipe": "openstl",
            "simvp_spatio_kernel_enc": 3,
            "simvp_spatio_kernel_dec": 3,
            "simvp_mlp_ratio": 8.0,
            "simvp_drop": 0.0,
            "simvp_drop_path": 0.0,
        }

        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "args": ckpt_args}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)

        infer_model, infer_cfg = infer_lib.build_model_from_saved_args(
            loaded["args"],
            image_mode="L",
            image_size=64,
            overrides={"simvp_model_type": "v2", "simvp_recipe": "openstl"},
        )
        predict_model, predict_cfg = predict_lib.build_model_from_ckpt_args(
            loaded["args"],
            image_mode="L",
            image_size=64,
            overrides={"simvp_model_type": "simvpv2"},
        )

        self.assertEqual(infer_cfg["simvp_model_type"], "gsta")
        self.assertEqual(infer_cfg["simvp_recipe"], "openstl")
        self.assertEqual(predict_cfg["simvp_model_type"], "gsta")
        self.assertEqual(predict_cfg["simvp_recipe"], "openstl")

        infer_model.load_state_dict(loaded["model"], strict=True)
        predict_model.load_state_dict(loaded["model"], strict=True)
        x = torch.randn(1, 8, 1, 64, 64)
        self.assertEqual(tuple(infer_model(x).shape), (1, 2, 1, 64, 64))
        self.assertEqual(tuple(predict_model(x).shape), (1, 2, 1, 64, 64))

    def test_canonical_choices_remain_small_and_stable(self):
        self.assertEqual(SIMVP_MODEL_TYPE_CHOICES, ("incepu", "gsta", "moganet"))
        self.assertIn("predformer_quadruplet_tsst", SUPPORTED_ARCHS)

    def test_build_forecast_model_kwargs_includes_predformer_facts_fields(self):
        model_kwargs, metadata = build_forecast_model_kwargs_from_config(
            {
                "arch": "predformer_facts",
                "in_T": 4,
                "out_T": 2,
                "predformer_patch_size": 8,
                "predformer_dim": 65,
                "predformer_heads": 4,
                "predformer_dim_head": 16,
                "predformer_dropout": 0.1,
                "predformer_attn_dropout": 0.2,
                "predformer_drop_path": 0.3,
                "predformer_scale_dim": 2,
                "predformer_depth": 3,
                "predformer_transformer_depth": 1,
            },
            image_mode="L",
            image_size=32,
        )
        self.assertEqual(metadata["arch"], "predformer_facts")
        self.assertEqual(model_kwargs["predformer_patch_size"], 8)
        self.assertEqual(model_kwargs["predformer_dim"], 65)
        self.assertEqual(model_kwargs["predformer_heads"], 4)
        self.assertEqual(model_kwargs["predformer_dim_head"], 16)
        self.assertEqual(model_kwargs["predformer_dropout"], 0.1)
        self.assertEqual(model_kwargs["predformer_attn_dropout"], 0.2)
        self.assertEqual(model_kwargs["predformer_drop_path"], 0.3)
        self.assertEqual(model_kwargs["predformer_scale_dim"], 2)
        self.assertEqual(model_kwargs["predformer_depth"], 3)
        self.assertEqual(model_kwargs["predformer_transformer_depth"], 1)

    def test_build_forecast_model_kwargs_allows_predformer_cli_overrides(self):
        model_kwargs, _ = build_forecast_model_kwargs_from_config(
            {
                "arch": "predformer_facts",
                "predformer_patch_size": 16,
                "predformer_dim": 256,
            },
            image_mode="L",
            image_size=32,
            overrides={
                "predformer_patch_size": 8,
                "predformer_dim": 65,
                "predformer_heads": 4,
                "predformer_dim_head": 16,
                "predformer_depth": 2,
                "predformer_transformer_depth": 2,
            },
        )
        self.assertEqual(model_kwargs["predformer_patch_size"], 8)
        self.assertEqual(model_kwargs["predformer_dim"], 65)
        self.assertEqual(model_kwargs["predformer_heads"], 4)
        self.assertEqual(model_kwargs["predformer_dim_head"], 16)
        self.assertEqual(model_kwargs["predformer_depth"], 2)
        self.assertEqual(model_kwargs["predformer_transformer_depth"], 2)

    def test_build_forecast_model_kwargs_supports_predformer_quadruplet_tsst(self):
        model_kwargs, metadata = build_forecast_model_kwargs_from_config(
            {
                "arch": "predformer_quadruplet_tsst",
                "in_T": 4,
                "out_T": 2,
                "predformer_patch_size": 8,
                "predformer_dim": 64,
                "predformer_heads": 4,
                "predformer_dim_head": 16,
                "predformer_depth": 2,
                "predformer_transformer_depth": 2,
            },
            image_mode="L",
            image_size=32,
        )
        self.assertEqual(metadata["arch"], "predformer_quadruplet_tsst")
        self.assertEqual(model_kwargs["predformer_transformer_depth"], 2)
        model = SimVPForecast(**model_kwargs)
        x = torch.randn(1, 4, 1, 32, 32)
        self.assertEqual(tuple(model(x).shape), (1, 2, 1, 32, 32))

    def test_build_forecast_model_kwargs_includes_predrnnv2_fields(self):
        model_kwargs, metadata = build_forecast_model_kwargs_from_config(
            {
                "arch": "predrnnv2",
                "in_T": 8,
                "out_T": 2,
                "predrnnv2_hidden": "8,8,8,8",
                "predrnnv2_filter_size": 3,
                "predrnnv2_patch_size": 2,
                "predrnnv2_stride": 1,
                "predrnnv2_layer_norm": True,
                "predrnnv2_decouple_beta": 0.25,
                "reverse_scheduled_sampling": True,
            },
            image_mode="L",
            image_size=32,
        )
        self.assertEqual(metadata["arch"], "predrnnv2")
        self.assertEqual(model_kwargs["predrnnv2_hidden"], "8,8,8,8")
        self.assertEqual(model_kwargs["predrnnv2_filter_size"], 3)
        self.assertEqual(model_kwargs["predrnnv2_patch_size"], 2)
        self.assertEqual(model_kwargs["predrnnv2_stride"], 1)
        self.assertTrue(model_kwargs["predrnnv2_layer_norm"])
        self.assertAlmostEqual(model_kwargs["predrnnv2_decouple_beta"], 0.25)
        self.assertTrue(model_kwargs["reverse_scheduled_sampling"])

    def test_build_forecast_model_kwargs_includes_mim_fields(self):
        model_kwargs, metadata = build_forecast_model_kwargs_from_config(
            {
                "arch": "mim",
                "in_T": 8,
                "out_T": 2,
                "mim_hidden": "8,8,8,8",
                "mim_filter_size": 3,
                "mim_patch_size": 2,
                "mim_stride": 1,
                "mim_layer_norm": True,
                "reverse_scheduled_sampling": True,
            },
            image_mode="L",
            image_size=32,
        )
        self.assertEqual(metadata["arch"], "mim")
        self.assertEqual(model_kwargs["mim_hidden"], "8,8,8,8")
        self.assertEqual(model_kwargs["mim_filter_size"], 3)
        self.assertEqual(model_kwargs["mim_patch_size"], 2)
        self.assertEqual(model_kwargs["mim_stride"], 1)
        self.assertTrue(model_kwargs["mim_layer_norm"])
        self.assertTrue(model_kwargs["reverse_scheduled_sampling"])


if __name__ == "__main__":
    unittest.main()
