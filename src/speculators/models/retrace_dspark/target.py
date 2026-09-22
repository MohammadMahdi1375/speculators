"""Local cached target and actual-trajectory vLLM hidden-state verification."""

from dataclasses import dataclass

import torch


@dataclass
class TargetStates:
    auxiliary: torch.Tensor
    scoring: torch.Tensor


class LocalTarget:
    def __init__(self, model, layer_ids):
        self.model = model.eval().requires_grad_(False)
        self.layer_ids = layer_ids
        self.cache = None
        self.tokens = []
        self.last = None
        self.calls = 0

    def reset(self):
        self.cache, self.tokens, self.last = None, [], None

    @torch.no_grad()
    def states(self, tokens):
        # Reuse only the committed prefix; verification suffixes are rolled back
        # even when the corrected token happens to match a later draft token.
        shared = 0
        for old, new in zip(self.tokens, tokens):
            if old != new:
                break
            shared += 1
        if (
            shared == len(tokens)
            and shared == len(self.tokens)
            and self.last is not None
        ):
            return self.last
        if shared == len(tokens):
            shared = max(0, shared - 1)
        if self.cache is not None:
            self.cache.crop(shared)
        ids = torch.tensor([tokens[shared:]], device=self.model.device)
        output = self.model.model(
            ids,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        aux = torch.cat([output.hidden_states[i] for i in self.layer_ids], -1)
        scoring = output.last_hidden_state
        if shared:
            aux = torch.cat((self.last.auxiliary[:, :shared], aux), 1)
            scoring = torch.cat((self.last.scoring[:, :shared], scoring), 1)
        self.last = TargetStates(aux.detach(), scoring.detach())
        self.tokens = list(tokens)
        self.cache = output.past_key_values
        self.calls += 1
        return self.last


class RemoteTarget:
    """Uses the existing connector with the draft's actual token IDs.

    Each round sends a complete prefix, so this is a correctness-oriented
    transport, not an incremental-KV training server. The inference engine uses
    native caches separately. File-transfer payloads remain full precision.
    """

    def __init__(self, draft, endpoint, model, expected_layers, timeout=180):
        from pathlib import Path

        from hs_connectors.transfer import FileTransfer
        from openai import OpenAI

        self.client = OpenAI(base_url=endpoint, api_key="EMPTY", timeout=timeout)
        self.transfer = FileTransfer(Path("/tmp/retrace-unused-cache"))
        self.draft, self.model = draft, model
        self.expected_layers, self.timeout = expected_layers, timeout
        self.calls = 0
        self.tokens, self.last = [], None

    def reset(self):
        self.tokens, self.last = [], None

    @torch.no_grad()
    def states(self, tokens):
        if tokens == self.tokens and self.last is not None:
            return self.last
        from speculators.data_generation.offline import check_hidden_states
        from speculators.data_generation.vllm_client import generate_hidden_states

        handle = generate_hidden_states(
            self.client, self.model, {"input_ids": list(tokens)}, timeout=self.timeout
        )
        try:
            payload = self.transfer.get_generated(handle)
            if payload is None:
                raise RuntimeError("vLLM returned no hidden-state payload")
            check_hidden_states(payload, tokens)
            hidden = payload["hidden_states"]
            expected = (len(tokens), self.expected_layers + 1, self.draft.hidden_size)
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"Hidden-state shape {tuple(hidden.shape)} != {expected}; check server target-layer-ids"
                )
            hidden = hidden.to(device=self.draft.device, dtype=self.draft.dtype)
            # The native connector's last slot is BEFORE the final target norm,
            # matching speculators.train.data + DFlashDraftModel._backbone_forward.
            self.last = TargetStates(
                hidden[:, :-1].flatten(1)[None],
                self.draft.verifier_norm(hidden[:, -1])[None],
            )
            self.tokens = list(tokens)
            self.calls += 1
            return self.last
        finally:
            self.transfer.delete(handle)
