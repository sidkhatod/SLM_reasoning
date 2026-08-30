"""
Layer 3 - Dual-Signal Adaptive Curriculum (DSAC).

Standard pipelines are curriculum-blind: they feed problems regardless of whether the
model can currently learn from them. Too easy collapses entropy; too hard yields no
actionable gradient. DSAC scores candidate problems on two live signals -

  * prefix validity   (VPPO-style): how confident the model is in its own opening steps
  * semantic uncertainty (SEED-GRPO-style): how much short sampled continuations diverge

- and keeps a buffer of the problems closest to the target band on both, i.e. the
model's productive learning zone. The buffer is refreshed every N steps so the
curriculum tracks the model as it improves rather than going stale.
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Reuse the frozen MiniLM embedder from the ICR layer
from training.layers.icr import get_embedder
from utils.metrics import record_curriculum_selection

# Defaults; every one is overridable from the YAML config.
DEFAULTS = {
    "dsac_target_validity": 0.5,
    "dsac_target_uncertainty": 0.5,
    "dsac_validity_weight": 1.0,
    "dsac_uncertainty_weight": 1.0,
    "dsac_buffer_size": 64,
    "dsac_candidate_pool": 256,
    "dsac_p_buffer": 0.8,
    "dsac_refresh_steps": 50,
    "dsac_probe_new_tokens": 20,
    "dsac_uncertainty_samples": 4,
    "dsac_uncertainty_new_tokens": 30,
}


def _cfg(config: dict, key: str):
    return (config or {}).get(key, DEFAULTS[key])


def compute_prefix_validity(model, tokenizer, prompt: str, partial_completions: list[str]) -> float:
    """
    Estimates how 'on track' the early reasoning tokens are.
    Computes the average normalized log probability (validity) of the partial completions.
    Lower probability roughly means the model is already lost early.

    Args:
        model: CausalLM model
        tokenizer: Model tokenizer
        prompt: The input prompt
        partial_completions: List of early reasoning step strings

    Returns:
        float: Average prefix validity score in [0.0, 1.0]
    """
    if not partial_completions:
        return 0.0

    was_training = model.training
    model.eval()
    validities = []

    try:
        with torch.no_grad():
            # Tokenize the prompt exactly the way the joint text will be tokenized,
            # otherwise the special-token handling shifts the boundary by one.
            prompt_len = len(tokenizer(prompt).input_ids)

            for partial in partial_completions:
                inputs = tokenizer(prompt + partial, return_tensors="pt").to(model.device)
                input_ids = inputs.input_ids[0]

                # If the prompt is somehow longer than the text, fail gracefully
                if prompt_len >= len(input_ids) or prompt_len < 1:
                    validities.append(0.0)
                    continue

                logits = model(**inputs).logits  # (1, seq_len, vocab_size)

                # Focus on the log probabilities of the *partial completion* tokens
                shift_logits = logits[0, prompt_len - 1:-1, :].contiguous()
                shift_labels = input_ids[prompt_len:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(shift_logits.float(), shift_labels)

                # Map CrossEntropy (NLL) to a [0,1] probability scale
                validities.append(float(torch.exp(-loss).item()))
    finally:
        if was_training:
            model.train()

    return float(np.mean(validities)) if validities else 0.0


def compute_semantic_uncertainty(model, tokenizer, prompt: str, num_samples: int = 4,
                                 max_new_tokens: int = 30) -> float:
    """
    Samples short continuations and estimates uncertainty via semantic divergence.
    High pairwise similarity between samples means the model is confident (low
    uncertainty); divergent samples mean it has not settled on an approach.
    """
    was_training = model.training
    model.eval()

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Generate short semantic trajectories
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        if was_training:
            model.train()

    prompt_len = inputs.input_ids.shape[1]
    samples = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    samples = [s for s in samples if s.strip()]
    if len(samples) <= 1:
        return 0.0

    # Embed using frozen MiniLM
    embedder = get_embedder()
    embeddings = embedder.encode(samples, convert_to_tensor=True, show_progress_bar=False)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    embeddings = embeddings.detach().cpu().float()

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(embeddings, embeddings.transpose(0, 1))

    n = embeddings.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    avg_sim = sim_matrix[mask].mean().item()

    # Uncertainty is inverse to similarity.
    # High similarity -> Low Uncertainty.
    uncertainty = 1.0 - avg_sim
    return max(0.0, min(1.0, uncertainty))


def score_problem(model, tokenizer, prompt: str, config: dict) -> float:
    """
    Combines prefix validity and semantic uncertainty into a 'productive zone' score.
    High scores indicate the problem is in the Goldilocks zone (not too easy, not too hard).
    """
    was_training = model.training
    model.eval()
    try:
        # Generate a single greedy prefix to measure validity against
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=_cfg(config, "dsac_probe_new_tokens"),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            partial = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        if was_training:
            model.train()

    if not partial.strip():
        return 0.0

    validity = compute_prefix_validity(model, tokenizer, prompt, [partial])
    uncertainty = compute_semantic_uncertainty(
        model, tokenizer, prompt,
        num_samples=_cfg(config, "dsac_uncertainty_samples"),
        max_new_tokens=_cfg(config, "dsac_uncertainty_new_tokens"),
    )

    target_validity = _cfg(config, "dsac_target_validity")
    target_uncertainty = _cfg(config, "dsac_target_uncertainty")
    weight_v = _cfg(config, "dsac_validity_weight")
    weight_u = _cfg(config, "dsac_uncertainty_weight")

    # Measure distance from ideal middle band.
    # The closer to target, the higher the score.
    dist_v = abs(validity - target_validity)
    dist_u = abs(uncertainty - target_uncertainty)

    denom = max(1e-8, weight_v + weight_u)
    score = 1.0 - ((weight_v * dist_v) + (weight_u * dist_u)) / denom
    return max(0.0, score)


def extract_prompt(item) -> str:
    """Flexible prompt extraction across the dataset shapes used in this repo."""
    if isinstance(item, dict):
        if "prompt" in item:
            return item["prompt"]
        if "question" in item:
            return item["question"]
        if "text" in item:
            # Naive extract if an SFT text string is present
            return item["text"].split("Answer:")[0] + "Answer:"
    return str(item)


class CurriculumBuffer:
    """Holds the current productive-zone problem set and serves samples from it."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.buffer = []
        self.last_refresh_step = -1

    @property
    def refresh_steps(self) -> int:
        return int(_cfg(self.config, "dsac_refresh_steps"))

    def refresh(self, model, tokenizer, full_dataset: list, buffer_size: int = None,
                sample_size: int = None):
        """
        Re-scores a random pool of the dataset and keeps the top-scoring problems.

        `sample_size` (the candidate pool) must exceed `buffer_size` for the curriculum
        to exert any selection pressure - otherwise every scored candidate lands in the
        buffer and DSAC degenerates to random sampling. The defaults (pool 256 -> buffer
        64) keep a 4:1 ratio; the assertion below stops a config from silently
        disabling the layer.
        """
        buffer_size = int(buffer_size if buffer_size is not None
                          else _cfg(self.config, "dsac_buffer_size"))
        sample_size = int(sample_size if sample_size is not None
                          else _cfg(self.config, "dsac_candidate_pool"))

        if sample_size <= buffer_size:
            print(f"[DSAC] Warning: candidate pool ({sample_size}) <= buffer size "
                  f"({buffer_size}); widening the pool to {buffer_size * 4} so the "
                  "curriculum actually selects.")
            sample_size = buffer_size * 4

        sample_size = min(sample_size, len(full_dataset))
        if sample_size == 0:
            self.buffer = []
            return

        print(f"\n[DSAC] Refreshing curriculum buffer (scoring {sample_size} candidates "
              f"-> keeping top {buffer_size})...")
        candidates = random.sample(list(full_dataset), sample_size)

        scored_candidates = []
        for item in candidates:
            score = score_problem(model, tokenizer, extract_prompt(item), self.config)
            scored_candidates.append((score, item))

        # Sort descending by productive zone score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Take the top N (those closest to the Goldilocks zone)
        kept = scored_candidates[:buffer_size]
        self.buffer = [item for _, item in kept]

        if kept:
            print(f"[DSAC] Buffer refreshed ({len(self.buffer)} problems). "
                  f"Top score: {kept[0][0]:.3f}, cutoff: {kept[-1][0]:.3f}, "
                  f"pool min: {scored_candidates[-1][0]:.3f}")

    def maybe_refresh(self, step: int, model, tokenizer, full_dataset: list) -> bool:
        """Refreshes on the configured cadence. Returns True if a refresh ran."""
        if step <= self.last_refresh_step:
            return False
        if step % self.refresh_steps != 0:
            return False
        self.refresh(model, tokenizer, full_dataset)
        self.last_refresh_step = step
        return True

    def sample_problem(self, full_dataset: list, p_buffer: float = None):
        """
        Samples a problem either from the productive buffer or randomly from the full
        dataset. The random share is deliberate: it keeps the curriculum from
        over-fitting to a stale buffer between refreshes.
        """
        if p_buffer is None:
            p_buffer = float(_cfg(self.config, "dsac_p_buffer"))

        if self.buffer and random.random() < p_buffer:
            record_curriculum_selection(from_buffer=True)
            return random.choice(self.buffer)

        record_curriculum_selection(from_buffer=False)
        return random.choice(list(full_dataset))


class RandomBuffer:
    """No-op stand-in used by the `use_dsac: false` ablations (uniform sampling)."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.buffer = []
        self.last_refresh_step = -1

    def refresh(self, *args, **kwargs):
        pass

    def maybe_refresh(self, *args, **kwargs):
        return False

    def sample_problem(self, full_dataset: list, p_buffer: float = None):
        record_curriculum_selection(from_buffer=False)
        return random.choice(list(full_dataset))
