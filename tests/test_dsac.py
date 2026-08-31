"""Layer 3 - Dual-Signal Adaptive Curriculum."""

import pytest

from training.layers.dsac import (
    CurriculumBuffer,
    RandomBuffer,
    extract_prompt,
    score_problem,
)
from utils.metrics import _tracker, productive_zone_ratio, reset_metrics

TOY_DATASET = [{"prompt": f"Question: problem {i}\nAnswer:", "answer": str(i)}
               for i in range(40)]


@pytest.fixture(autouse=True)
def _clean_metrics():
    reset_metrics()
    yield
    reset_metrics()


class FakeScorer:
    """Stands in for the policy: `score_problem` is monkeypatched to read this map."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def __call__(self, model, tokenizer, prompt, config):
        self.calls += 1
        return self.scores.get(prompt, 0.0)


def test_extract_prompt_handles_every_dataset_shape():
    assert extract_prompt({"prompt": "P"}) == "P"
    assert extract_prompt({"question": "Q"}) == "Q"
    assert extract_prompt({"text": "Question: X\nAnswer: Y"}).endswith("Answer:")
    assert extract_prompt("raw string") == "raw string"


def test_buffer_keeps_the_highest_scoring_problems(monkeypatch):
    # Score i/40 so the top 10 are problems 30-39.
    scores = {item["prompt"]: i / 40 for i, item in enumerate(TOY_DATASET)}
    scorer = FakeScorer(scores)
    monkeypatch.setattr("training.layers.dsac.score_problem", scorer)

    buf = CurriculumBuffer({"dsac_buffer_size": 10, "dsac_candidate_pool": 40})
    buf.refresh(model=None, tokenizer=None, full_dataset=TOY_DATASET)

    assert len(buf.buffer) == 10
    assert scorer.calls == 40
    kept = {item["prompt"] for item in buf.buffer}
    assert kept == {item["prompt"] for item in TOY_DATASET[30:]}


def test_candidate_pool_is_widened_when_it_cannot_select(monkeypatch):
    """
    A pool <= buffer size means every scored candidate is kept, which silently turns
    DSAC into uniform sampling. The layer must widen the pool instead.
    """
    monkeypatch.setattr("training.layers.dsac.score_problem",
                        FakeScorer({item["prompt"]: i for i, item in enumerate(TOY_DATASET)}))

    buf = CurriculumBuffer({})
    buf.refresh(None, None, TOY_DATASET, buffer_size=8, sample_size=8)
    assert len(buf.buffer) == 8
    # 8 kept out of a widened 32-candidate pool = real selection pressure.
    kept = {item["prompt"] for item in buf.buffer}
    assert kept != {item["prompt"] for item in TOY_DATASET[:8]}


def test_refresh_fires_on_the_configured_cadence(monkeypatch):
    calls = []
    monkeypatch.setattr(CurriculumBuffer, "refresh",
                        lambda self, *a, **k: calls.append(self.last_refresh_step))

    buf = CurriculumBuffer({"dsac_refresh_steps": 50})
    for step in range(1, 151):
        buf.maybe_refresh(step, None, None, TOY_DATASET)

    assert len(calls) == 3, "should refresh at steps 50, 100 and 150"
    assert buf.last_refresh_step == 150


def test_sampling_prefers_the_buffer_and_records_the_productive_zone_ratio():
    buf = CurriculumBuffer({"dsac_p_buffer": 1.0})
    buf.buffer = TOY_DATASET[:5]

    picks = [buf.sample_problem(TOY_DATASET) for _ in range(50)]
    assert all(p in TOY_DATASET[:5] for p in picks)
    assert productive_zone_ratio(_tracker.curriculum_selection_log) == 1.0

    reset_metrics()
    buf.config["dsac_p_buffer"] = 0.0
    for _ in range(50):
        buf.sample_problem(TOY_DATASET)
    assert productive_zone_ratio(_tracker.curriculum_selection_log) == 0.0


def test_empty_buffer_falls_back_to_the_full_dataset():
    buf = CurriculumBuffer({"dsac_p_buffer": 1.0})
    assert buf.buffer == []
    assert buf.sample_problem(TOY_DATASET) in TOY_DATASET
    assert productive_zone_ratio(_tracker.curriculum_selection_log) == 0.0


def test_random_buffer_ablation_never_reports_productive_selections():
    buf = RandomBuffer({})
    for _ in range(20):
        assert buf.sample_problem(TOY_DATASET) in TOY_DATASET
    assert buf.maybe_refresh(50, None, None, TOY_DATASET) is False
    assert productive_zone_ratio(_tracker.curriculum_selection_log) == 0.0


def test_productive_zone_score_peaks_at_the_target_band(monkeypatch):
    """A problem sitting exactly at both targets is maximally productive."""
    monkeypatch.setattr("training.layers.dsac.compute_prefix_validity",
                        lambda *a, **k: 0.5)
    monkeypatch.setattr("training.layers.dsac.compute_semantic_uncertainty",
                        lambda *a, **k: 0.5)

    class StubModel:
        training = False
        device = "cpu"

        def eval(self): pass

        def train(self): pass

        def generate(self, **kwargs):
            import torch
            return torch.tensor([[1, 2, 3, 4]])

    class StubTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, **kwargs):
            import torch

            class E(dict):
                input_ids = torch.tensor([[1, 2]])

                def to(self, device):
                    return self

            enc = E()
            enc["input_ids"] = E.input_ids
            return enc

        def decode(self, ids, **kwargs):
            return " some partial reasoning"

    config = {"dsac_target_validity": 0.5, "dsac_target_uncertainty": 0.5}
    assert score_problem(StubModel(), StubTokenizer(), "Q", config) == pytest.approx(1.0)

    # A problem the model is certain about (too easy) scores far lower.
    monkeypatch.setattr("training.layers.dsac.compute_prefix_validity", lambda *a, **k: 1.0)
    monkeypatch.setattr("training.layers.dsac.compute_semantic_uncertainty", lambda *a, **k: 0.0)
    assert score_problem(StubModel(), StubTokenizer(), "Q", config) == pytest.approx(0.5)
