from __future__ import annotations

from typing import Any, Mapping, Optional


SIMVP_MODEL_TYPE_CHOICES = ("incepu", "gsta", "moganet")
SIMVP_MODEL_TYPE_ALIASES = ("incepu", "gsta", "moga", "moganet", "v1", "v2", "simvpv1", "simvpv2")
SIMVP_RECIPE_CHOICES = ("auto", "simvp", "openstl")
PREDRNNPP_RECIPE_CHOICES = ("simvp", "openstl")
SIMVP_OPENSTL_MODEL_TYPES = frozenset(("gsta", "moganet"))

SIMVP_OPENSTL_TRAIN_PRESET = {
    "hid_S": 64,
    "hid_T": 512,
    "N_S": 4,
    "N_T": 8,
    "lr": 1e-3,
    "batch_size": 16,
    "simvp_drop_path": 0.0,
    "opt": "adam",
    "sched": "onecycle",
}

_SIMVP_MODEL_TYPE_ALIAS_MAP = {
    "incepu": "incepu",
    "gsta": "gsta",
    "moga": "moganet",
    "moganet": "moganet",
    "v1": "incepu",
    "simvpv1": "incepu",
    "v2": "gsta",
    "simvpv2": "gsta",
}


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def normalize_simvp_model_type(model_type: Optional[str]) -> str:
    if model_type is None:
        return "incepu"
    normalized = str(model_type).strip().lower()
    if normalized in _SIMVP_MODEL_TYPE_ALIAS_MAP:
        return _SIMVP_MODEL_TYPE_ALIAS_MAP[normalized]
    raise ValueError(
        f"Unsupported SimVP model_type '{model_type}'. "
        f"Canonical choices: {SIMVP_MODEL_TYPE_CHOICES}; aliases: {SIMVP_MODEL_TYPE_ALIASES}."
    )


def normalize_simvp_recipe(recipe: Optional[str]) -> str:
    if recipe is None:
        return "auto"
    normalized = str(recipe).strip().lower()
    if normalized in SIMVP_RECIPE_CHOICES:
        return normalized
    raise ValueError(f"Unsupported SimVP recipe '{recipe}'. Available choices: {SIMVP_RECIPE_CHOICES}.")


def normalize_predrnnpp_recipe(recipe: Optional[str]) -> str:
    normalized = str("simvp" if recipe is None else recipe).strip().lower()
    if normalized in PREDRNNPP_RECIPE_CHOICES:
        return normalized
    raise ValueError(
        f"Unsupported PredRNN++ recipe '{recipe}'. Available choices: {PREDRNNPP_RECIPE_CHOICES}."
    )


def get_effective_simvp_recipe(
    arch: Optional[str],
    model_type: Optional[str],
    recipe: Optional[str],
) -> str:
    normalized_recipe = normalize_simvp_recipe(recipe)
    normalized_arch = str("simvp" if arch is None else arch).strip().lower()
    normalized_model_type = normalize_simvp_model_type(model_type)
    if normalized_arch != "simvp":
        return normalized_recipe
    if normalized_recipe == "auto":
        return "openstl" if normalized_model_type in SIMVP_OPENSTL_MODEL_TYPES else "simvp"
    return normalized_recipe


def is_simvp_openstl_recipe(
    arch: Optional[str],
    model_type: Optional[str],
    recipe: Optional[str],
) -> bool:
    normalized_arch = str("simvp" if arch is None else arch).strip().lower()
    normalized_model_type = normalize_simvp_model_type(model_type)
    return (
        normalized_arch == "simvp"
        and normalized_model_type in SIMVP_OPENSTL_MODEL_TYPES
        and get_effective_simvp_recipe(normalized_arch, normalized_model_type, recipe) == "openstl"
    )


def is_simvp_gsta_openstl_recipe(
    arch: Optional[str],
    model_type: Optional[str],
    recipe: Optional[str],
) -> bool:
    normalized_arch = str("simvp" if arch is None else arch).strip().lower()
    normalized_model_type = normalize_simvp_model_type(model_type)
    return (
        normalized_arch == "simvp"
        and normalized_model_type == "gsta"
        and get_effective_simvp_recipe(normalized_arch, normalized_model_type, recipe) == "openstl"
    )


def build_forecast_model_kwargs_from_config(
    config: Mapping[str, Any],
    *,
    image_mode: str,
    image_size: int,
    overrides: Optional[Mapping[str, Any]] = None,
):
    overrides = {} if overrides is None else dict(overrides)
    channels = 1 if image_mode == "L" else 3

    in_T = int(_coalesce(overrides.get("in_T"), config.get("in_T"), 8))
    out_T = int(_coalesce(overrides.get("out_T"), config.get("out_T"), 2))
    arch = str(_coalesce(overrides.get("arch"), config.get("arch"), "simvp")).strip().lower()
    simvp_model_type = normalize_simvp_model_type(
        _coalesce(overrides.get("simvp_model_type"), config.get("simvp_model_type"), "incepu")
    )
    simvp_recipe = normalize_simvp_recipe(
        _coalesce(overrides.get("simvp_recipe"), config.get("simvp_recipe"), "auto")
    )
    predrnnpp_recipe = normalize_predrnnpp_recipe(
        _coalesce(overrides.get("predrnnpp_recipe"), config.get("predrnnpp_recipe"), "simvp")
    )
    use_local_branch = bool(_coalesce(overrides.get("use_local_branch"), config.get("use_local_branch"), False))
    local_top = int(_coalesce(overrides.get("local_top"), config.get("local_top"), 186))
    local_bottom = int(_coalesce(overrides.get("local_bottom"), config.get("local_bottom"), 410))
    local_crop = (local_top, local_bottom)
    earthfarseer_depth = int(_coalesce(overrides.get("earthfarseer_depth"), config.get("earthfarseer_depth"), 12))
    earthfarseer_spatial_depth = _coalesce(
        overrides.get("earthfarseer_spatial_depth"),
        config.get("earthfarseer_spatial_depth"),
        None,
    )
    earthfarseer_temporal_depth = _coalesce(
        overrides.get("earthfarseer_temporal_depth"),
        config.get("earthfarseer_temporal_depth"),
        None,
    )

    kwargs = {
        "in_T": in_T,
        "out_T": out_T,
        "C": channels,
        "H": image_size,
        "W": image_size,
        "hid_S": int(config.get("hid_S", 32)),
        "hid_T": int(config.get("hid_T", 128)),
        "N_S": int(config.get("N_S", 4)),
        "N_T": int(config.get("N_T", 4)),
        "simvp_model_type": simvp_model_type,
        "simvp_recipe": simvp_recipe,
        "simvp_spatio_kernel_enc": int(config.get("simvp_spatio_kernel_enc", 3)),
        "simvp_spatio_kernel_dec": int(config.get("simvp_spatio_kernel_dec", 3)),
        "simvp_mlp_ratio": float(config.get("simvp_mlp_ratio", 8.0)),
        "simvp_drop": float(config.get("simvp_drop", 0.0)),
        "simvp_drop_path": float(config.get("simvp_drop_path", 0.0)),
        "tau_spatio_kernel_enc": int(config.get("tau_spatio_kernel_enc", 3)),
        "tau_spatio_kernel_dec": int(config.get("tau_spatio_kernel_dec", 3)),
        "tau_mlp_ratio": float(config.get("tau_mlp_ratio", 8.0)),
        "tau_drop": float(config.get("tau_drop", 0.0)),
        "tau_drop_path": float(config.get("tau_drop_path", 0.0)),
        "earthfarseer_incep_ker": str(config.get("earthfarseer_incep_ker", "3,5,7,11")),
        "earthfarseer_groups": int(config.get("earthfarseer_groups", 8)),
        "earthfarseer_num_interactions": int(config.get("earthfarseer_num_interactions", 3)),
        "earthfarseer_patch_size": int(config.get("earthfarseer_patch_size", 16)),
        "earthfarseer_embed_dim": int(config.get("earthfarseer_embed_dim", 768)),
        "earthfarseer_depth": earthfarseer_depth,
        "earthfarseer_spatial_depth": None if earthfarseer_spatial_depth is None else int(earthfarseer_spatial_depth),
        "earthfarseer_temporal_depth": None
        if earthfarseer_temporal_depth is None
        else int(earthfarseer_temporal_depth),
        "earthfarseer_mlp_ratio": float(config.get("earthfarseer_mlp_ratio", 4.0)),
        "earthfarseer_drop": float(config.get("earthfarseer_drop", 0.0)),
        "earthfarseer_drop_path": float(config.get("earthfarseer_drop_path", 0.0)),
        "convlstm_hidden": str(config.get("convlstm_hidden", "128,128,128,128")),
        "convlstm_filter_size": int(config.get("convlstm_filter_size", 5)),
        "convlstm_patch_size": int(config.get("convlstm_patch_size", 4)),
        "convlstm_stride": int(config.get("convlstm_stride", 1)),
        "convlstm_layer_norm": bool(config.get("convlstm_layer_norm", False)),
        "predrnnpp_hidden": str(config.get("predrnnpp_hidden", "128,128,128,128")),
        "predrnnpp_filter_size": int(config.get("predrnnpp_filter_size", 5)),
        "predrnnpp_patch_size": int(config.get("predrnnpp_patch_size", 4)),
        "predrnnpp_stride": int(config.get("predrnnpp_stride", 1)),
        "predrnnpp_layer_norm": bool(config.get("predrnnpp_layer_norm", False)),
        "predrnnpp_recipe": predrnnpp_recipe,
        "predrnnpp_reverse_scheduled_sampling": bool(config.get("reverse_scheduled_sampling", False)),
        "predrnnv2_hidden": str(config.get("predrnnv2_hidden", "128,128,128,128")),
        "predrnnv2_filter_size": int(config.get("predrnnv2_filter_size", 5)),
        "predrnnv2_patch_size": int(config.get("predrnnv2_patch_size", 4)),
        "predrnnv2_stride": int(config.get("predrnnv2_stride", 1)),
        "predrnnv2_layer_norm": bool(config.get("predrnnv2_layer_norm", False)),
        "predrnnv2_decouple_beta": float(config.get("predrnnv2_decouple_beta", 0.1)),
        "predrnnv2_reverse_scheduled_sampling": bool(config.get("reverse_scheduled_sampling", False)),
        "predformer_patch_size": int(_coalesce(overrides.get("predformer_patch_size"), config.get("predformer_patch_size"), 16)),
        "predformer_dim": int(_coalesce(overrides.get("predformer_dim"), config.get("predformer_dim"), 256)),
        "predformer_heads": int(_coalesce(overrides.get("predformer_heads"), config.get("predformer_heads"), 8)),
        "predformer_dim_head": int(
            _coalesce(overrides.get("predformer_dim_head"), config.get("predformer_dim_head"), 32)
        ),
        "predformer_dropout": float(
            _coalesce(overrides.get("predformer_dropout"), config.get("predformer_dropout"), 0.0)
        ),
        "predformer_attn_dropout": float(
            _coalesce(overrides.get("predformer_attn_dropout"), config.get("predformer_attn_dropout"), 0.0)
        ),
        "predformer_drop_path": float(
            _coalesce(overrides.get("predformer_drop_path"), config.get("predformer_drop_path"), 0.0)
        ),
        "predformer_scale_dim": int(
            _coalesce(overrides.get("predformer_scale_dim"), config.get("predformer_scale_dim"), 4)
        ),
        # predformer_depth keeps the public config name; in this FacTS port it is the
        # shared stack depth used by both the temporal and spatial transformer branches.
        "predformer_depth": int(_coalesce(overrides.get("predformer_depth"), config.get("predformer_depth"), 4)),
        "arch": arch,
        "hybrid_depth": int(config.get("hybrid_depth", 2)),
        "hybrid_heads": int(config.get("hybrid_heads", 8)),
        "hybrid_ffn_ratio": float(config.get("hybrid_ffn_ratio", 4.0)),
        "hybrid_attn_dropout": float(config.get("hybrid_attn_dropout", 0.1)),
        "hybrid_ffn_dropout": float(config.get("hybrid_ffn_dropout", 0.1)),
        "hybrid_drop_path": float(config.get("hybrid_drop_path", 0.1)),
        "use_local_branch": use_local_branch,
        "local_crop": local_crop,
    }
    metadata = {
        "in_T": in_T,
        "out_T": out_T,
        "arch": arch,
        "simvp_model_type": simvp_model_type,
        "simvp_recipe": simvp_recipe,
        "simvp_recipe_effective": get_effective_simvp_recipe(arch, simvp_model_type, simvp_recipe),
        "predrnnpp_recipe": predrnnpp_recipe,
        "earthfarseer_depth": earthfarseer_depth,
        "earthfarseer_spatial_depth": None if earthfarseer_spatial_depth is None else int(earthfarseer_spatial_depth),
        "earthfarseer_temporal_depth": None
        if earthfarseer_temporal_depth is None
        else int(earthfarseer_temporal_depth),
        "use_local_branch": use_local_branch,
        "local_crop": local_crop,
    }
    return kwargs, metadata
