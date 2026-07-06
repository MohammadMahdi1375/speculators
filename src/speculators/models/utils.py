import warnings
import json
from pathlib import Path
from functools import partial

import torch
from transformers import AutoConfig, PretrainedConfig


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func




# DFLASH_CFG_SHIM: Transformers in this environment may not yet recognize
# model_type="deepseek_v4".  For training we only need config attributes such
# as vocab_size, hidden_size, num_attention_heads, num_hidden_layers, etc.;
# fall back to a lightweight attribute-access wrapper around config.json.
class _DFlashConfigShim(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self):
        return dict(self)


def _dflash_wrap_config(value):
    if isinstance(value, dict):
        return _DFlashConfigShim({k: _dflash_wrap_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dflash_wrap_config(v) for v in value]
    return value


def _dflash_load_raw_config(verifier_name_or_path: str, cause: Exception):
    config_path = Path(verifier_name_or_path) / "config.json"
    if not config_path.exists():
        raise cause
    warnings.warn(
        "AutoConfig could not load verifier config; falling back to raw "
        f"config.json for DSV4 compatibility: {config_path}. Original error: {cause}",
        stacklevel=2,
    )
    return _dflash_wrap_config(json.loads(config_path.read_text()))


def get_verifier_config(verifier_name_or_path: str) -> PretrainedConfig:
    try:
        verifier_config = AutoConfig.from_pretrained(
            verifier_name_or_path,
            trust_remote_code=True,
        )
    except Exception as exc:
        verifier_config = _dflash_load_raw_config(verifier_name_or_path, exc)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
) -> list[int]:
    if target_layer_ids is not None:
        return target_layer_ids

    num_layers = get_verifier_config(verifier_name_or_path).num_hidden_layers
    target_layer_ids = [2, num_layers // 2, num_layers - 3]
    warnings.warn(
        DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
        stacklevel=3,
    )
    return target_layer_ids


def resolve_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Resolve a dense draft MLP ``intermediate_size`` from a verifier config.

    The draft is an independent small *dense* decoder, so its FFN width is a design
    choice rather than something to reconcile with the verifier's routed capacity:

    * Dense verifiers expose ``intermediate_size`` directly; the draft mirrors it.
    * MoE verifiers have no dense ``intermediate_size`` (their FFN is a routed set of
      small experts), so the draft falls back to the widely used ``3 * hidden_size``
      gated-MLP ratio -- the Qwen3 dense convention that the dflash draft decoder
      follows. Pass ``--draft-config`` to set it explicitly instead.

    :raises ValueError: when the verifier config exposes neither ``intermediate_size``
        nor ``hidden_size`` (degenerate config; pass ``--draft-config``).
    """
    dense = getattr(verifier_config, "intermediate_size", None)
    if dense is not None:
        return int(dense)

    hidden_size = getattr(verifier_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Verifier config exposes neither `intermediate_size` nor `hidden_size`, "
            "so a draft intermediate_size cannot be inferred. Pass --draft-config to "
            "set the draft architecture explicitly."
        )

    intermediate_size = 3 * int(hidden_size)
    warnings.warn(
        "Verifier config has no dense intermediate_size (likely MoE); using draft "
        f"intermediate_size={intermediate_size} (3 x hidden_size = {hidden_size}). "
        "Pass --draft-config to override.",
        stacklevel=3,
    )
    return intermediate_size


# DFLASH_CFG_SHIM_V2: Transformers in this env may not recognize
# model_type="deepseek_v4".  Override get_verifier_config with a fallback that
# reads local config.json and exposes dict keys as attributes.
class _DFlashConfigShimV2(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self):
        return dict(self)


def _dflash_wrap_config_v2(value):
    if isinstance(value, dict):
        return _DFlashConfigShimV2({k: _dflash_wrap_config_v2(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dflash_wrap_config_v2(v) for v in value]
    return value


def get_verifier_config(verifier_name_or_path: str):  # type: ignore[override]
    try:
        verifier_config = AutoConfig.from_pretrained(
            verifier_name_or_path,
            trust_remote_code=True,
        )
    except Exception as exc:
        import json as _json
        import warnings as _warnings
        from pathlib import Path as _Path

        config_path = _Path(verifier_name_or_path) / "config.json"
        if not config_path.exists():
            raise
        _warnings.warn(
            "AutoConfig could not load verifier config; falling back to raw "
            f"config.json for DSV4 compatibility: {config_path}. Original error: {exc}",
            stacklevel=2,
        )
        verifier_config = _dflash_wrap_config_v2(_json.loads(config_path.read_text()))

    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config
