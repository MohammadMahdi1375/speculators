"""Device setup and independent ReTrace model construction."""

import copy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

from .config import ReTraceSpeculatorConfig
from .core import ReTraceDraftModel
from .execution import fuse_target_norms, set_draft_attention
from .target import LocalTarget, RemoteTarget


def device_for(name):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(name)
    return torch.device(name)


def setup(args, device):
    attention_backend = getattr(args, "attention_backend", "eager")
    target_norm = getattr(args, "target_norm", "reference")
    if args.vllm_endpoint and target_norm != "reference":
        raise ValueError("Fused target norms apply only to the local target")
    config = AutoConfig.from_pretrained(args.target, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("Expected the dense Qwen3 target")
    if args.draft:
        saved = ReTraceSpeculatorConfig.from_pretrained(
            args.draft, local_files_only=True
        )
        if saved.speculators_model_type != "retrace":
            raise ValueError(
                "Use an isolated ReTrace checkpoint; legacy DSpARK+ReTrace checkpoints have a different architecture"
            )
        saved.speculators_config.verifier.name_or_path = args.target
        draft = ReTraceDraftModel.from_pretrained(
            args.draft, config=saved, local_files_only=True
        )
    else:
        layers = args.target_layer_ids or [1, 9, 17, 25, 33]
        if (
            not layers
            or min(layers) < 1
            or max(layers) >= config.num_hidden_layers
            or len(set(layers)) != len(layers)
        ):
            raise ValueError(
                "Target layer IDs must be unique HF hidden-state indices in [1, num_hidden_layers)"
            )
        draft_config = copy.deepcopy(config)
        draft_config.num_hidden_layers = args.num_layers
        draft_config.layer_types = ["full_attention"] * args.num_layers
        draft_config._attn_implementation = "eager"
        draft_config.attention_dropout = 0.0
        saved = ReTraceSpeculatorConfig(
            transformer_layer_config=draft_config,
            draft_vocab_size=config.vocab_size,
            block_size=args.block_size,
            aux_hidden_state_layer_ids=layers,
            mask_token_id=args.mask_token_id,
            retrace_warmup_steps=args.retrace_warmup_steps,
            speculators_config=SpeculatorsConfig(
                algorithm="retrace",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=args.block_size - 1)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(args.target),
            ),
        )
        draft = ReTraceDraftModel(saved)
        draft.load_verifier_weights()
    if (
        draft.hidden_size != config.hidden_size
        or draft.verifier_vocab_size != config.vocab_size
    ):
        raise ValueError("Checkpoint and target dimensions differ")
    if not 0 <= draft.mask_token_id < config.vocab_size:
        raise ValueError("Mask token must be an existing target embedding row")
    set_draft_attention(draft, attention_backend)
    draft = draft.to(device=device, dtype=getattr(torch, args.dtype))
    if args.vllm_endpoint:
        print(f"Target backend: vLLM ({args.vllm_endpoint}); trainer loads draft and "
              "frozen shared embeddings/head/norm, without target transformer layers", flush=True)
        target = RemoteTarget(
            draft,
            args.vllm_endpoint,
            args.served_model_name or args.target,
            len(draft.target_layer_ids),
            args.request_timeout,
        )
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.target,
                local_files_only=True,
                dtype=getattr(torch, args.dtype),
                attn_implementation=attention_backend,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        if target_norm == "npu":
            count = fuse_target_norms(model)
            print(f"Frozen target: {count} fused NPU RMSNorm modules", flush=True)
        draft.embed_tokens = model.model.embed_tokens
        draft.lm_head = model.lm_head
        draft.verifier_lm_head = model.lm_head
        draft.verifier_norm = model.model.norm
        target = LocalTarget(model, draft.target_layer_ids)
    try:
        eos = GenerationConfig.from_pretrained(
            args.target, local_files_only=True
        ).eos_token_id
    except OSError:
        eos = config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    return draft, target, eos


def add_model_args(parser):
    parser.add_argument("--target", "--verifier-name-or-path", required=True)
    parser.add_argument("--draft", "--from-pretrained")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--attention-backend", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument(
        "--target-norm", choices=("reference", "npu"), default="reference"
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--target-layer-ids", type=int, nargs="+")
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--retrace-warmup-steps", type=int, default=50)
    parser.add_argument("--vllm-endpoint", default="")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--request-timeout", type=float, default=180)


def sync_device(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
