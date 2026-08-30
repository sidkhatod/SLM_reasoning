"""
Layer 1 - Semantic Prefix Masking (SPM) + the NTHR bonus.

Negative gradient updates on incorrect completions can accidentally suppress tokens
that are actually correct reasoning steps, simply because those tokens appear as a
shared prefix before the incorrect completion diverges. This causes "Lazy Likelihood
Displacement" (LLD), where the model unlearns valid reasoning patterns. SPM detects
these shared prefixes and dampens the negative gradient on them, so the model is
penalised only for the tokens where it actually went wrong.

The core operates on *token id sequences* rather than strings: those are what the
policy loss is actually applied to, and re-encoding a decoded string does not
round-trip to the same length, which would silently misalign the mask.
`apply_spm_from_text` is a thin convenience wrapper for offline/unit-test use.
"""

import torch


def _shared_prefix_positions(target_ids, reference_id_lists) -> set:
    """Longest-common-prefix positions between `target_ids` and ANY reference sequence."""
    shared = set()
    for ref_ids in reference_id_lists:
        max_len = min(len(target_ids), len(ref_ids))
        for i in range(max_len):
            if target_ids[i] == ref_ids[i]:
                shared.add(i)
            else:
                # Break at the first divergence for strict prefix matching
                break
    return shared


def find_shared_prefix_tokens(correct_completions: list[str], incorrect_completions: list[str],
                              tokenizer) -> list[set[int]]:
    """
    Identifies token positions where the reasoning content overlaps between a group's
    correct and incorrect completions.

    Args:
        correct_completions: List of string completions that are labeled as correct.
        incorrect_completions: List of string completions that are labeled as incorrect.
        tokenizer: The tokenizer used for the model.

    Returns:
        A list of sets, one for each incorrect completion. Each set contains the token
        indices in that incorrect completion that are part of a shared prefix with ANY
        correct completion.

    Note: a future extension can swap the exact token-id match below for a semantic
    similarity check over token windows, to catch paraphrased-but-equivalent steps.
    """
    correct_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in correct_completions]
    incorrect_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in incorrect_completions]
    return [_shared_prefix_positions(inc, correct_tokens) for inc in incorrect_tokens]


def find_shared_correct_tokens(correct_completions: list[str], incorrect_completions: list[str],
                               tokenizer) -> list[set[int]]:
    """
    Companion to find_shared_prefix_tokens. Identifies which tokens in the CORRECT
    completions are shared with the incorrect ones, so we can apply the NTHR bonus.
    """
    correct_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in correct_completions]
    incorrect_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in incorrect_completions]
    return [_shared_prefix_positions(cor, incorrect_tokens) for cor in correct_tokens]


def compute_soft_mask(shared_token_positions: set[int], sequence_length: int,
                      mask_strength: float = 0.5) -> torch.Tensor:
    """
    Returns a per-token multiplier in [0,1] to apply to the negative gradient at
    shared prefix positions.

    Why soft masking:
    Instead of hard-masking (zeroing out) the gradient on shared tokens, soft masking
    multiplies the negative advantage by a dampening factor (e.g., 0.5). This approach
    preserves some learning signal in case the shared tokens are actually suboptimal,
    while mitigating severe Lazy Likelihood Displacement.

    Args:
        shared_token_positions: Set of integer token indices that are shared.
        sequence_length: Total number of tokens in the completion.
        mask_strength: How much to dampen the gradient (0.0 = no masking, 1.0 = full masking).

    Returns:
        A 1D tensor of shape (sequence_length,) containing multipliers.
    """
    # Initialize multipliers to 1.0 (no dampening)
    mask = torch.ones(sequence_length, dtype=torch.float32)

    # Apply dampening to shared positions
    multiplier = 1.0 - mask_strength
    for pos in shared_token_positions:
        if 0 <= pos < sequence_length:
            mask[pos] = multiplier

    return mask


def apply_nthr_bonus(shared_correct_token_positions: set[int], sequence_length: int,
                     bonus_strength: float = 0.1) -> torch.Tensor:
    """
    The NTHR extension: gives a small POSITIVE signal to tokens that are shared
    AND come from a correct completion, on top of the masking above.

    Why this is needed:
    If a correct completion and an incorrect completion share a valid reasoning prefix,
    the model might still slightly penalize the correct prefix from the incorrect trajectory.
    NTHR counters this by slightly boosting the reward for those specific shared tokens
    in the successful trajectory.

    Args:
        shared_correct_token_positions: Set of integer token indices in the correct completion.
        sequence_length: Total number of tokens in the completion.
        bonus_strength: The additive bonus to apply to the advantage.

    Returns:
        A 1D tensor of shape (sequence_length,) containing the additive bonuses.
    """
    bonus = torch.zeros(sequence_length, dtype=torch.float32)
    for pos in shared_correct_token_positions:
        if 0 <= pos < sequence_length:
            bonus[pos] = bonus_strength

    return bonus


def compute_lld_severity(group_token_ids: list[list[int]], correctness_labels: list[bool]) -> float:
    """
    LLD Severity Score: the fraction of tokens in incorrect completions that sit inside
    a prefix shared with at least one correct completion. Those are exactly the tokens a
    vanilla GRPO negative gradient would corrupt.

    Returns 0.0 for degenerate groups (all-correct or all-incorrect), where no
    correct/incorrect contrast exists.
    """
    correct = [ids for ids, ok in zip(group_token_ids, correctness_labels) if ok]
    incorrect = [ids for ids, ok in zip(group_token_ids, correctness_labels) if not ok]

    if not correct or not incorrect:
        return 0.0

    total_tokens = 0
    total_shared = 0
    for inc in incorrect:
        total_tokens += len(inc)
        total_shared += len(_shared_prefix_positions(inc, correct))

    if total_tokens == 0:
        return 0.0
    return float(total_shared / total_tokens)


def apply_spm(group_token_ids: list[list[int]],
              group_advantages,
              config: dict = None,
              correctness_labels: list[bool] = None) -> list[torch.Tensor]:
    """
    Ties SPM and NTHR together. Called by grpo_train.py once per generation group.

    Args:
        group_token_ids: G lists of the *generated* token ids (padding/EOS already trimmed).
        group_advantages: G scalar advantages (list, or a 1D tensor).
        config: dict reading `spm_mask_strength` (default 0.5) and
                `nthr_bonus_strength` (default 0.1).
        correctness_labels: optional G booleans. When omitted, a completion counts as
                "correct" if its advantage is > 0 (GRPO advantages are group-normalised,
                so > 0 means better than the group average). Passing real outcome labels
                is preferred - it makes the prefix contrast reflect actual correctness
                rather than relative reward.

    Returns:
        A list of G 1D tensors of token-level advantages, each aligned 1:1 with the
        corresponding entry of `group_token_ids`.
    """
    config = config or {}
    mask_strength = float(config.get("spm_mask_strength", 0.5))
    bonus_strength = float(config.get("nthr_bonus_strength", 0.1))

    if isinstance(group_advantages, torch.Tensor):
        advantages = [float(a) for a in group_advantages.detach().flatten().tolist()]
    else:
        advantages = [float(a) for a in group_advantages]

    if correctness_labels is None:
        correctness_labels = [a > 0 for a in advantages]

    correct_ids = [ids for ids, ok in zip(group_token_ids, correctness_labels) if ok]
    incorrect_ids = [ids for ids, ok in zip(group_token_ids, correctness_labels) if not ok]

    modified_advantages = []

    for i, token_ids in enumerate(group_token_ids):
        seq_len = len(token_ids)
        # Start with a constant token-level advantage for the whole sequence
        token_advantages = torch.full((seq_len,), advantages[i], dtype=torch.float32)

        if seq_len == 0:
            modified_advantages.append(token_advantages)
            continue

        if correctness_labels[i]:
            # Apply the NTHR bonus on the prefix this correct trajectory shares with
            # the failing ones, countering residual suppression from their updates.
            if incorrect_ids:
                shared_pos = _shared_prefix_positions(token_ids, incorrect_ids)
                token_advantages = token_advantages + apply_nthr_bonus(
                    shared_pos, seq_len, bonus_strength
                )
        else:
            # Apply SPM: dampen the (negative) advantage on the shared valid prefix.
            if correct_ids:
                shared_pos = _shared_prefix_positions(token_ids, correct_ids)
                token_advantages = token_advantages * compute_soft_mask(
                    shared_pos, seq_len, mask_strength
                )

        modified_advantages.append(token_advantages)

    return modified_advantages


def apply_spm_from_text(group_completions: list[str],
                        group_advantages,
                        tokenizer,
                        config: dict = None,
                        correctness_labels: list[bool] = None) -> list[torch.Tensor]:
    """String-input convenience wrapper around `apply_spm` (offline analysis / tests)."""
    group_token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in group_completions]
    return apply_spm(group_token_ids, group_advantages, config, correctness_labels)
