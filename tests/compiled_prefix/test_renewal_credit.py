import itertools

import pytest
import torch

from speculators.models.dflash_prefix.renewal_credit import dense_anchors, round_saving_credit


def brute_rounds(accepted, known, start, repair=None, span=0):
    end = start
    while end < len(known) and known[end]:
        end += 1
    rounds, index = 0, start
    while index < end:
        rounds += 1
        step = accepted[index] + 1 + (span if index == repair else 0)
        index += step
    return rounds


def test_round_credit_matches_complete_counterfactual_simulation():
    # Tests bonus +1, segment boundaries, all starting anchors, and a repaired
    # first round followed by the unmodified current policy.
    for accepted in itertools.product(range(3), repeat=4):
        for known in ([True] * 4, [True, True, False, True], [False] * 4):
            spans = torch.tensor([1, 2, 0, 1])
            delta, eligible, rounds = round_saving_credit(torch.tensor(accepted), spans, torch.tensor(known))
            for i in range(4):
                before = brute_rounds(accepted, known, i)
                after = brute_rounds(accepted, known, i, i, spans[i].item())
                assert rounds[i] == before
                assert delta[i] == (before - after if eligible[i] else 0)
                assert not delta.requires_grad


def test_equal_local_tau_gain_can_save_different_round_counts():
    # Both repair spans are one token. In A, moving to anchor 2 reaches a
    # strong block. In B, anchor 1 was strong, and the same move bypasses it.
    accepted_a = torch.tensor([0, 0, 3, 0, 0, 0, 0])
    accepted_b = torch.tensor([0, 4, 0, 0, 0, 0, 0])
    spans = torch.tensor([1, 0, 0, 0, 0, 0, 0])
    known = torch.ones(7, dtype=torch.bool)
    a, _, _ = round_saving_credit(accepted_a, spans, known)
    b, _, _ = round_saving_credit(accepted_b, spans, known)
    assert a[0] == 1
    assert b[0] == -3
    # Training clamps the supplementary credit to >=0. It never rewards an
    # intentional wrong token merely to reach an easier next anchor.


def test_artificial_window_end_cannot_create_round_saving_credit():
    # Baseline: 0->1->3, repair: 0->2->3. They only meet at unobserved end 3.
    delta, eligible, _ = round_saving_credit(torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0]), torch.ones(3, dtype=torch.bool))
    assert not eligible[0] and delta[0] == 0


def test_dense_windows_are_contiguous_reproducible_and_fit_backbone_shape():
    mask = torch.zeros(1, 30)
    mask[:, 7:24] = 1
    torch.manual_seed(42)
    a, valid = dense_anchors(mask, 8, 4)
    torch.manual_seed(42)
    b, other_valid = dense_anchors(mask, 8, 4)
    assert torch.equal(a, b) and torch.equal(valid, other_valid)
    assert (a[1:] - a[:-1]).eq(1).all()
    assert a[-1] + 4 < 30
    assert torch.equal(valid, mask[0, a].bool())
    with pytest.raises(ValueError, match="fit"):
        dense_anchors(mask, 28, 4)


def test_no_known_labels_produces_zero_round_credit():
    delta, eligible, rounds = round_saving_credit(torch.tensor([2, 3, 0]), torch.tensor([1, 1, 1]), torch.zeros(3, dtype=torch.bool))
    assert not eligible.any() and not delta.any() and not rounds.any()
