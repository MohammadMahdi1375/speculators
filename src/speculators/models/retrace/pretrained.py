"""Strict, local-only import of the public Qwen3 DFlash drafter.

Only ReTrace's three conditioning matrices are newly initialized. Every draft
backbone tensor must be present, finite, and exactly preserved by the import.
No remote Python from the checkpoint is executed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

from .config import ReTraceSpeculatorConfig
from .core import ReTraceDraftModel

SHARED = {
    "embed_tokens.weight",
    "lm_head.weight",
    "verifier_lm_head.weight",
    "verifier_norm.weight",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    result = []
    for name in names:
        if Path(name).name != name or not (path / name).is_file():
            raise ValueError(f"Missing or invalid safetensors shard: {name}")
        result.append(path / name)
    return result


def source_config(source, target):
    raw = json.loads((Path(source) / "config.json").read_text())
    target_config = AutoConfig.from_pretrained(target, local_files_only=True)
    if target_config.model_type != "qwen3":
        raise ValueError("This importer requires the dense Qwen3 target")
    kind = raw.get("speculators_model_type")
    if kind == "dflash":
        config = DFlashSpeculatorConfig.from_pretrained(source, local_files_only=True)
        tl = config.transformer_layer_config
        layer_ids = config.aux_hidden_state_layer_ids
        block_size, mask_id = config.block_size, config.mask_token_id
        convention = "speculators HF hidden-state indices, copied unchanged"
    elif kind is None and raw.get("architectures") == ["DFlashDraftModel"]:
        if raw.get("model_type") != "qwen3" or "dflash_config" not in raw:
            raise ValueError("Expected the official dense Qwen3 DFlash format")
        drop = {
            "architectures",
            "auto_map",
            "block_size",
            "dflash_config",
            "num_target_layers",
        }
        tl = AutoConfig.for_model(
            "qwen3", **{k: v for k, v in raw.items() if k not in drop | {"model_type"}}
        )
        layer_ids = [int(i) + 1 for i in raw["dflash_config"]["target_layer_ids"]]
        block_size = int(raw["block_size"])
        mask_id = int(raw["dflash_config"]["mask_token_id"])
        convention = "z-lab decoder-layer indices + 1 = HF hidden-state indices"
    else:
        raise ValueError(
            "Use an original DFlash checkpoint, not DFlash2, DSpARK or ReTrace"
        )
    for name in (
        "hidden_size",
        "vocab_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
    ):
        if getattr(tl, name) != getattr(target_config, name):
            raise ValueError(f"Draft/target architecture mismatch: {name}")
    if (
        not layer_ids
        or len(set(layer_ids)) != len(layer_ids)
        or not all(0 < i < target_config.num_hidden_layers for i in layer_ids)
    ):
        raise ValueError(
            "Auxiliary features must be distinct intermediate target layers"
        )
    if not 0 <= mask_id < target_config.vocab_size or block_size < 2:
        raise ValueError("Invalid mask token or block size")
    if raw.get("sample_from_anchor", False):
        raise ValueError("DFlash import must preserve the verified anchor")
    tl._attn_implementation = "eager"
    tl.attention_dropout = 0.0
    common = dict(
        transformer_layer_config=tl,
        aux_hidden_state_layer_ids=layer_ids,
        draft_vocab_size=target_config.vocab_size,
        block_size=block_size,
        mask_token_id=mask_id,
        sample_from_anchor=False,
    )
    verifier = VerifierConfig.from_pretrained(target)
    config = ReTraceSpeculatorConfig(
        **common,
        training_objective="clean_block_ce",
        pretrained_source=str(Path(source).absolute()),
        speculators_config=SpeculatorsConfig(
            algorithm="retrace",
            default_proposal_method="greedy",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
            ],
            verifier=verifier,
        ),
    )
    return config, convention


def inspect_source(source, target):
    config, convention = source_config(source, target)
    files = checkpoint_files(source)
    keys = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in keys:
                    raise ValueError(f"Duplicate tensor across shards: {name}")
                keys[name] = list(handle.get_slice(name).get_shape())
    hidden = config.transformer_layer_config.hidden_size
    if keys.get("fc.weight") != [
        hidden,
        hidden * len(config.aux_hidden_state_layer_ids),
    ]:
        raise ValueError(
            "Feature fusion shape does not match the checkpoint's layer IDs"
        )
    return config, {
        "source": str(Path(source).absolute()),
        "layer_index_convention": convention,
        "aux_hidden_state_layer_ids": config.aux_hidden_state_layer_ids,
        "block_size": config.block_size,
        "proposals_per_round": config.block_size - 1,
        "draft_layers": config.transformer_layer_config.num_hidden_layers,
        "tensor_shapes": keys,
    }


def load_dflash(source, target, dtype=torch.bfloat16, load_shared=True):
    config, report = inspect_source(source, target)
    # Reproducible new conditioning matrices; never reinitialize loaded tensors.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        draft = ReTraceDraftModel(config).to(dtype=dtype)
    expected = {
        name
        for name in draft.state_dict()
        if name not in SHARED and not name.startswith("retrace.")
    }
    loaded = set()
    with torch.no_grad():
        parameters = draft.state_dict()
        for path in checkpoint_files(source):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in SHARED:
                        continue
                    if name not in expected:
                        raise ValueError(f"Unexpected source tensor: {name}")
                    value = handle.get_tensor(name)
                    if (
                        value.shape != parameters[name].shape
                        or not value.is_floating_point()
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError(
                            f"Invalid tensor shape, dtype or values: {name}"
                        )
                    parameters[name].copy_(value)
                    if not torch.equal(parameters[name], value.to(dtype)):
                        raise ValueError(
                            f"Pretrained tensor changed during import: {name}"
                        )
                    loaded.add(name)
    if loaded != expected:
        raise ValueError(
            f"Missing pretrained backbone tensors: {sorted(expected - loaded)}"
        )
    if load_shared:
        draft.load_verifier_weights()
    hashes = {
        p.name: sha256(p)
        for p in [Path(source) / "config.json", *checkpoint_files(source)]
    }
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    draft.config.pretrained_fingerprint = fingerprint
    report.update(
        conditioning_initialization_seed=42,
        source_sha256=hashes,
        fingerprint=fingerprint,
        imported_tensors=len(loaded),
        backbone_tensors_exact_after_dtype_cast=True,
        new_tensors=[
            name for name in draft.state_dict() if name.startswith("retrace.")
        ],
        residual_projection_is_zero=not bool(
            torch.count_nonzero(draft.retrace.value.weight)
        ),
    )
    return draft, report


def prepare(source, target, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Use a new initialization directory: {output}")
    draft, report = load_dflash(source, target)
    output.mkdir(parents=True)
    draft.save_pretrained(output / "retrace")
    raw = draft.config.to_dict()
    for key in list(raw):
        if key.startswith("retrace_") or key in {
            "training_objective",
            "pretrained_source",
            "pretrained_fingerprint",
        }:
            raw.pop(key)
    raw["speculators_model_type"] = "dflash"
    raw["architectures"] = ["DFlashSpeculator"]
    raw["speculators_config"]["algorithm"] = "dflash"
    baseline = DFlashSpeculatorConfig(**raw)
    baseline_dir = output / "dflash"
    baseline_dir.mkdir()
    baseline.save_pretrained(baseline_dir)
    weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in draft.state_dict().items()
        if name not in SHARED and not name.startswith("retrace.")
    }
    save_file(
        weights, str(baseline_dir / "model.safetensors"), metadata={"format": "pt"}
    )
    (output / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _, result = inspect_source(args.source, args.target)
        result["tensor_count"] = len(result.pop("tensor_shapes"))
    else:
        if not args.output:
            parser.error("--output is required unless --check-only is used")
        result = prepare(args.source, args.target, args.output)
        result.pop("tensor_shapes")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
