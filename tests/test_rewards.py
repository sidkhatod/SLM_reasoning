"""The 6-component reward function."""

import pytest

from rewards.reward_functions import (
    DEFAULT_WEIGHTS,
    _consistency_reward,
    _format_reward,
    _outcome_reward,
    _self_correction_reward,
    answers_match,
    compute_total_reward,
    extract_final_answer,
    normalize_answer,
    resolve_weights,
)

CFG = {"use_prm": False, "use_icr": True}


def test_answer_extraction_takes_the_marker_line_only():
    assert extract_final_answer("reasoning\n\nFinal Answer: 42\nsome trailing noise") == "42"
    assert extract_final_answer("final answer: yes") == "yes"
    assert extract_final_answer("no marker here") == ""


@pytest.mark.parametrize("raw,expected", [
    ("18", "18"), ("18.0", "18"), ("1,200", "1200"), ("$1200.00", "1200"),
    (" 72 ", "72"), ("-5", "-5"), ("3.5", "3.5"),
    ("Yes", "yes"), ("TRUE", "yes"), ("no", "no"), ("False", "no"),
    ("C", "C"), ("(C)", "C"), ("c)", "C"),
])
def test_answer_normalisation_covers_all_three_answer_shapes(raw, expected):
    assert normalize_answer(raw) == expected


def test_formatting_differences_do_not_count_as_wrong_answers():
    """Exact string matching failed GSM8K answers over '$', ',' and trailing zeros."""
    assert answers_match("$1,200.00", "1200")
    assert answers_match("The answer is 72", "72")
    assert not answers_match("71", "72")
    assert not answers_match("", "72")
    # An empty ground truth can never be satisfied.
    assert not answers_match("72", "")


def test_outcome_reward_is_binary_and_verifiable():
    completions = ["...\nFinal Answer: 18", "...\nFinal Answer: 19", "no marker"]
    assert _outcome_reward(completions, "18") == [1.0, 0.0, 0.0]


def test_format_reward_is_earnable_from_the_completion_alone():
    """
    'Let's think step by step.' lives in the *prompt*, so scoring it here made the
    component permanently unearnable.
    """
    good = "We add 4 and 6 to get 10, then halve it.\n\nFinal Answer: 5"
    assert _format_reward([good]) == [1.0]

    assert _format_reward(["Final Answer: 5"]) == [0.5], "no visible reasoning"
    assert _format_reward(["Some long reasoning that goes on for a while here."]) == [0.5]
    assert _format_reward(["Final Answer: 5\nFinal Answer: 6"])[0] < 1.0, \
        "duplicate markers are a reward-hacking pattern"
    assert _format_reward(["short"]) == [0.0]


def test_consistency_reward_matches_normalised_values_in_the_reasoning_tail():
    aligned = "We compute 6 * 200 = 1200 in total.\n\nFinal Answer: 1,200"
    detached = "We compute 6 * 200 = 1200 in total.\n\nFinal Answer: 47"
    assert _consistency_reward([aligned]) == [1.0]
    assert _consistency_reward([detached]) == [0.0]
    assert _consistency_reward(["no marker at all"]) == [0.0]


def test_self_correction_reward_detects_structured_revision():
    assert _self_correction_reward(["Wait, let me reconsider that step."]) == [1.0]
    assert _self_correction_reward(["Let me double-check the arithmetic."]) == [1.0]
    assert _self_correction_reward(["But wait, that gives a negative count."]) == [1.0]
    assert _self_correction_reward(["Straightforward: 2 + 2 = 4."]) == [0.0]


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_format_weight_stays_capped_against_reward_hacking():
    assert DEFAULT_WEIGHTS["format"] == 0.05
    assert DEFAULT_WEIGHTS["outcome"] > sum(
        v for k, v in DEFAULT_WEIGHTS.items() if k in ("format", "self_correction", "consistency")
    ), "outcome must dominate the heuristic components"


def test_disabled_component_weight_is_redistributed_not_dropped():
    """Otherwise the reward silently shrinks to [0, 0.8] and ablations stop comparing."""
    weights = resolve_weights({"use_prm": False, "reward_weights": DEFAULT_WEIGHTS})
    assert weights["prm"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["outcome"] > DEFAULT_WEIGHTS["outcome"]

    both_off = resolve_weights({"use_prm": False, "use_icr": False})
    assert both_off["prm"] == 0.0 and both_off["consensus"] == 0.0
    assert sum(both_off.values()) == pytest.approx(1.0)

    kept = resolve_weights({"use_prm": False, "redistribute_disabled_reward_weight": False})
    assert sum(kept.values()) == pytest.approx(0.8)


def test_total_reward_is_bounded_and_ranks_the_best_completion_first():
    completions = [
        "We add 4 and 6 to get 10. Let me double-check: 10 / 2 = 5.\n\nFinal Answer: 5",
        "We add 4 and 6 to get 10.\n\nFinal Answer: 20",
        "gibberish",
    ]
    group_context = {"icr_rewards": [0.9, 0.3, 0.1]}
    totals, components = compute_total_reward("Question: ...", completions, "5",
                                              group_context, CFG)

    assert len(totals) == 3
    assert all(0.0 <= t <= 1.0 for t in totals)
    assert totals[0] > totals[1] > totals[2]
    assert components["outcome"] == pytest.approx(1 / 3)
    assert set(components) == set(DEFAULT_WEIGHTS) | {"total"}


def test_missing_icr_context_scores_neutral_rather_than_penalising():
    totals, components = compute_total_reward("Q", ["...\nFinal Answer: 5"], "5", {}, CFG)
    assert components["consensus"] == 0.5
    assert 0.0 <= totals[0] <= 1.0


def test_empty_completion_group_does_not_crash():
    totals, components = compute_total_reward("Q", [], "5", {}, CFG)
    assert totals == []
    assert components["total"] == 0.0
