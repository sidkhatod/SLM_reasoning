"""Layer 2 - Implicit Consensus Rewards."""

import torch

from training.layers.icr import (
    cluster_and_align_steps,
    compute_consensus_reward,
    compute_icr_rewards,
    embed_steps,
    implicit_verification_score,
    split_into_steps,
)

# Two correct completions share the "divide by two" step; the incorrect one multiplies.
CORRECT_1 = "First we add 4 to 6. This gives 10. Then we divide by 2.\nFinal Answer: 5"
CORRECT_2 = "Add four to six to get 10. Divide both sides by two.\nFinal Answer: 5"
INCORRECT = "Add 4 and 6 to get 10. Then we multiply by 2.\nFinal Answer: 20"


def test_steps_split_on_sentences_not_decimal_points():
    steps = split_into_steps("The rate is 3.5 per hour. He works 2 hours. So 7.0 total.")
    assert len(steps) == 3, f"a decimal point must not split a step: {steps}"
    assert steps[0] == "The rate is 3.5 per hour."


def test_final_answer_line_is_excluded_from_steps():
    """It is scored by the outcome reward; keeping it would create a useless cluster."""
    steps = split_into_steps(CORRECT_1)
    assert all("Final Answer" not in s for s in steps)
    assert any("divide by 2" in s for s in steps)


def test_every_step_is_terminated_for_stable_embeddings():
    for s in split_into_steps("one step\nanother step"):
        assert s.endswith((".", "?", "!"))


def test_embed_steps_shape_and_empty_case():
    assert embed_steps([]).shape == (0, 384)
    emb = embed_steps(["A first step.", "A second step."])
    assert emb.shape == (2, 384)
    assert isinstance(emb, torch.Tensor)


def test_discriminative_steps_score_above_shared_ones():
    completions = [CORRECT_1, CORRECT_2, INCORRECT]
    labels = [True, True, False]
    embeddings = [embed_steps(split_into_steps(c)) for c in completions]

    clusters, assignments = cluster_and_align_steps(embeddings, labels,
                                                    similarity_threshold=0.5)
    assert clusters is not None
    assert len(assignments) == 3

    for cdata in clusters.values():
        assert -1.0 <= cdata["discriminativeness"] <= 1.0

    # A step present in every completion carries no information either way.
    universal = [c for c in clusters.values() if len(c["completions_containing"]) == 3]
    for c in universal:
        assert abs(c["discriminativeness"]) < 1e-9

    # At least one step must separate the correct completions from the incorrect one.
    assert max(c["discriminativeness"] for c in clusters.values()) > 0.0


def test_consensus_rewards_are_one_per_completion_and_bounded():
    completions = [CORRECT_1, CORRECT_2, INCORRECT]
    labels = [True, True, False]
    embeddings = [embed_steps(split_into_steps(c)) for c in completions]
    clusters, assignments = cluster_and_align_steps(embeddings, labels, 0.5)

    rewards = compute_consensus_reward(clusters, assignments, num_completions=3)
    assert isinstance(rewards, list) and len(rewards) == 3
    assert all(0.0 <= r <= 1.0 for r in rewards), "must share the [0,1] scale of the other components"
    assert (rewards[0] + rewards[1]) / 2 > rewards[2], \
        "correct completions should out-score the incorrect one on consensus"


def test_all_negative_group_falls_back_to_neutral():
    """All 8 completions wrong -> step alignment is impossible; outcome must drive it."""
    completions = [INCORRECT, INCORRECT, CORRECT_1]
    embeddings = [embed_steps(split_into_steps(c)) for c in completions]

    clusters, assignments = cluster_and_align_steps(embeddings, [False, False, False], 0.5)
    assert clusters is None and assignments is None

    rewards = compute_consensus_reward(clusters, assignments, num_completions=3)
    assert rewards == [0.5, 0.5, 0.5], "neutral, not zero - zero would be a penalty"


def test_all_positive_group_also_falls_back():
    embeddings = [embed_steps(split_into_steps(c)) for c in (CORRECT_1, CORRECT_2)]
    clusters, _ = cluster_and_align_steps(embeddings, [True, True], 0.5)
    assert clusters is None, "with no negative class nothing is discriminative"


def test_ivs_counts_only_strongly_discriminative_clusters():
    fake = {0: {"discriminativeness": 0.9}, 1: {"discriminativeness": 0.2},
            2: {"discriminativeness": -0.5}}
    assert implicit_verification_score(fake, threshold=0.5) == 1
    assert implicit_verification_score(fake, threshold=0.1) == 2
    assert implicit_verification_score(None) == 0


def test_end_to_end_entry_point_returns_rewards_and_telemetry():
    rewards, stats = compute_icr_rewards(
        [CORRECT_1, CORRECT_2, INCORRECT], [True, True, False],
        {"icr_similarity_threshold": 0.5},
    )
    assert len(rewards) == 3 and all(0.0 <= r <= 1.0 for r in rewards)
    assert stats["fallback"] is False
    assert stats["num_clusters"] > 0
    assert isinstance(stats["discriminative_clusters"], int)

    rewards, stats = compute_icr_rewards([INCORRECT] * 3, [False] * 3)
    assert rewards == [0.5, 0.5, 0.5]
    assert stats["fallback"] is True
    assert stats["discriminative_clusters"] == 0
