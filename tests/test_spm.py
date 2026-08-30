"""Layer 1 - Semantic Prefix Masking + NTHR."""

import torch

from conftest import WordTokenizer
from training.layers.spm import (
    apply_nthr_bonus,
    apply_spm,
    apply_spm_from_text,
    compute_lld_severity,
    compute_soft_mask,
    find_shared_prefix_tokens,
)

SHARED_PREFIX = "By order of operations we do multiplication first 3*4=12 Then 2+12="
CORRECT = f"{SHARED_PREFIX} 14."
# Same reasoning, diverges only at the final arithmetic step.
INCORRECT_SHARED = f"{SHARED_PREFIX} 10."
# Wrong from the first token - nothing to protect.
INCORRECT_DISJOINT = "We do addition first 2+3=5 5*4=20."


def test_shared_prefix_covers_the_common_reasoning_and_stops_at_divergence():
    tok = WordTokenizer()
    shared = find_shared_prefix_tokens([CORRECT], [INCORRECT_SHARED, INCORRECT_DISJOINT], tok)

    prefix_len = len(tok.encode(SHARED_PREFIX))
    assert shared[0] == set(range(prefix_len)), "the whole shared prefix must be detected"
    assert prefix_len not in shared[0], "divergent final token must NOT be masked"
    assert shared[1] == set(), "a completion sharing nothing must get an empty mask"


def test_soft_mask_dampens_only_shared_positions():
    mask = compute_soft_mask({0, 1, 2}, sequence_length=6, mask_strength=0.5)
    assert mask.tolist() == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    # mask_strength 1.0 is hard masking; 0.0 disables the layer.
    assert compute_soft_mask({0}, 2, 1.0).tolist() == [0.0, 1.0]
    assert compute_soft_mask({0}, 2, 0.0).tolist() == [1.0, 1.0]


def test_nthr_bonus_is_additive_and_positional():
    bonus = apply_nthr_bonus({1, 3}, sequence_length=4, bonus_strength=0.1)
    assert torch.allclose(bonus, torch.tensor([0.0, 0.1, 0.0, 0.1]))


def test_out_of_range_positions_are_ignored():
    assert compute_soft_mask({99}, 3, 0.5).tolist() == [1.0, 1.0, 1.0]
    assert apply_nthr_bonus({99}, 3, 0.1).tolist() == [0.0, 0.0, 0.0]


def test_spm_dampens_the_negative_gradient_on_shared_tokens():
    """The core claim: shared valid reasoning is penalised less than the divergent tail."""
    tok = WordTokenizer()
    completions = [CORRECT, INCORRECT_SHARED, INCORRECT_DISJOINT]
    advantages = [1.0, -1.0, -1.0]

    token_advs = apply_spm_from_text(
        completions, advantages, tok,
        config={"spm_mask_strength": 0.5, "nthr_bonus_strength": 0.1},
        correctness_labels=[True, False, False],
    )

    assert len(token_advs) == 3
    for adv, comp in zip(token_advs, completions):
        assert adv.shape[0] == len(tok.encode(comp))

    prefix_len = len(tok.encode(SHARED_PREFIX))

    shared_incorrect = token_advs[1]
    assert torch.allclose(shared_incorrect[:prefix_len], torch.full((prefix_len,), -0.5)), \
        "shared prefix of a wrong completion must be dampened to half the penalty"
    assert shared_incorrect[prefix_len].item() == -1.0, \
        "the token where it actually went wrong keeps the full penalty"

    assert torch.allclose(token_advs[2], torch.full_like(token_advs[2], -1.0)), \
        "a completion sharing no prefix is penalised in full"

    correct = token_advs[0]
    assert torch.allclose(correct[:prefix_len], torch.full((prefix_len,), 1.1)), \
        "NTHR must add a positive signal on the correct completion's shared prefix"
    assert correct[prefix_len].item() == 1.0


def test_spm_falls_back_to_advantage_signs_without_labels():
    tok = WordTokenizer()
    token_advs = apply_spm_from_text([CORRECT, INCORRECT_SHARED], [1.0, -1.0], tok)
    prefix_len = len(tok.encode(SHARED_PREFIX))
    assert token_advs[1][:prefix_len].max().item() < 0
    assert token_advs[1][0].item() > -1.0


def test_degenerate_groups_are_left_untouched():
    """All-correct or all-incorrect groups have no contrast, so nothing is masked."""
    tok = WordTokenizer()
    all_wrong = apply_spm_from_text(
        [INCORRECT_SHARED, INCORRECT_DISJOINT], [-0.5, -0.5], tok,
        correctness_labels=[False, False],
    )
    for adv in all_wrong:
        assert torch.allclose(adv, torch.full_like(adv, -0.5))

    empty = apply_spm([[]], [1.0], {}, correctness_labels=[True])
    assert empty[0].numel() == 0


def test_lld_severity_measures_the_corrupted_token_fraction():
    tok = WordTokenizer()
    ids = [tok.encode(c) for c in (CORRECT, INCORRECT_SHARED, INCORRECT_DISJOINT)]

    severity = compute_lld_severity(ids, [True, False, False])
    prefix_len = len(tok.encode(SHARED_PREFIX))
    expected = prefix_len / (len(ids[1]) + len(ids[2]))
    assert abs(severity - expected) < 1e-9
    assert 0.0 < severity < 1.0

    # No correct/incorrect contrast -> nothing can be displaced.
    assert compute_lld_severity(ids, [False, False, False]) == 0.0
    assert compute_lld_severity(ids, [True, True, True]) == 0.0
