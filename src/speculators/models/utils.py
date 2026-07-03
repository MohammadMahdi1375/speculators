import warnings

from transformers import AutoConfig, PretrainedConfig


def get_verifier_config(verifier_name_or_path: str) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
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

# ---------------------------------------------------------------------------
# Compatibility shim for fork modules kept by the dsv4 no-deletion overlay.
# DSpark/MTP/PEagle import this from speculators.models.utils.
# ---------------------------------------------------------------------------
def conditional_torch_compile(func=None, **compile_kwargs):
    """Conditionally apply torch.compile, otherwise return the object unchanged.

    Supports both:
        @conditional_torch_compile
        def f(...): ...

    and:
        @conditional_torch_compile(...)
        def f(...): ...
    """

    def _decorator(obj):
        import os

        if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
            return obj
        if os.environ.get("TORCHDYNAMO_DISABLE") == "1":
            return obj

        try:
            import torch
            torch_compile = getattr(torch, "compile", None)
            if torch_compile is None:
                return obj
            return torch_compile(obj, **compile_kwargs)
        except Exception:
            return obj

    if func is None:
        return _decorator

    return _decorator(func)

