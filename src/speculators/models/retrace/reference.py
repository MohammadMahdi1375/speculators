"""Shape-consistent, slow reference verification for numerical debugging.

The prompt is prefilled normally. Subsequent target tokens are evaluated one at
a time, including rejected proposals. Greedy acceptance and ReTrace memory are
unchanged. This control is not used by training or native vLLM inference.
"""

import torch

from .target import LocalTarget


class TokenwiseLocalTarget(LocalTarget):
    @torch.no_grad()
    def states(self, tokens):
        if not tokens:
            raise ValueError("A nonempty target prefix is required")
        if self.cache is None:
            # Same prompt prefill as the ordinary target-only greedy reference.
            return super().states(tokens)
        shared = 0
        for old, new in zip(self.tokens, tokens):
            if old != new:
                break
            shared += 1
        if shared == len(tokens):
            # An exact repeat returns the cached states. A shortened prefix
            # crops and recomputes its final token with one input token.
            return super().states(tokens)
        for end in range(shared + 1, len(tokens) + 1):
            result = super().states(tokens[:end])
        return result

    @staticmethod
    def project(head, hidden):
        """Use the same [1,H] head shape as single-request greedy decoding."""
        if hidden.ndim == 2 and hidden.shape[0] == 1:
            return head(hidden)
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1:
            raise ValueError("Tokenwise control supports one nonempty trajectory")
        return torch.stack(
            [head(hidden[:, index]) for index in range(hidden.shape[1])], dim=1
        )
