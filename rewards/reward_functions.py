"""
The 6-component PRISM-GRPO reward function.

    0.40 outcome         - final-answer correctness (verifiable)
    0.20 prm             - Math-Shepherd step-level quality
    0.20 consensus       - the implicit group signal from Layer 2 (ICR)
    0.10 consistency     - does the final answer follow from the stated reasoning
    0.05 self_correction - structured self-correction present
    0.05 format          - adherence to the SFT chain-of-thought format

Every component returns a score in [0, 1] so the weighted sum is also in [0, 1].
Components that are switched off (e.g. `use_prm: false` when no PRM fits in VRAM)
have their weight redistributed across the remaining components rather than
silently shrinking the reward scale.
"""

import gc
import os
import re

import numpy as np
import torch

# To avoid keeping the PRM in memory forever if not needed, we use a singleton pattern.
_prm_model = None
_prm_tokenizer = None
_prm_candidate_token_ids = None
_prm_step_tag_id = None
_prm_load_failed = False

# Math-Shepherd's own conventions: each reasoning step is terminated with the
# step tag, and the PRM's judgement is read off the logits at that position.
PRM_MODEL_ID = "peiyi9979/math-shepherd-mistral-7b-prm"
PRM_GOOD_TOKEN = "+"
PRM_BAD_TOKEN = "-"
PRM_STEP_TAG = "ки"


def _load_prm_model(model_id: str = PRM_MODEL_ID):
    """Loads Math-Shepherd in 4-bit. Returns True on success, False if unavailable."""
    global _prm_model, _prm_tokenizer, _prm_candidate_token_ids, _prm_step_tag_id, _prm_load_failed

    if _prm_model is not None:
        return True
    if _prm_load_failed:
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[Rewards] Loading Math-Shepherd PRM ({model_id}) in 4-bit...")
        _prm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        if _prm_tokenizer.pad_token is None:
            _prm_tokenizer.pad_token = _prm_tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _prm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        _prm_model.eval()

        # encode(...)[1:] drops the leading BOS that Mistral's tokenizer prepends.
        _prm_candidate_token_ids = _prm_tokenizer.encode(
            f"{PRM_GOOD_TOKEN} {PRM_BAD_TOKEN}"
        )[1:]
        _prm_step_tag_id = _prm_tokenizer.encode(f"{PRM_STEP_TAG}")[-1]
        return True
    except Exception as e:
        print(f"[Rewards] Could not load the PRM ({e}). "
              "Falling back to a neutral PRM score of 0.5; set `use_prm: false` "
              "in the config to redistribute its weight instead.")
        _prm_load_failed = True
        _prm_model = None
        _prm_tokenizer = None
        return False


def unload_prm_model():
    """Unloads the PRM to free up GPU memory."""
    global _prm_model, _prm_tokenizer
    if _prm_model is not None:
        del _prm_model
        del _prm_tokenizer
        _prm_model = None
        _prm_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Rewards] Unloaded Math-Shepherd PRM from memory.")


# Kept as a private alias for backwards compatibility with earlier call sites.
_unload_prm_model = unload_prm_model


def _tag_steps_for_prm(completion: str) -> str:
    """Rewrites a completion into Math-Shepherd's expected `Step N: ... ки` form."""
    body = re.split(r'Final Answer:', completion, flags=re.IGNORECASE)[0]
    lines = [ln.strip() for ln in body.split('\n') if ln.strip()]
    if not lines:
        lines = [completion.strip() or "(empty)"]
    return " ".join(f"Step {i + 1}: {ln} {PRM_STEP_TAG}" for i, ln in enumerate(lines))


def _prm_reward(prompt: str, completions: list[str], unload_after: bool = False,
                model_id: str = PRM_MODEL_ID) -> list[float]:
    """
    Computes step-level quality scores using Math-Shepherd.

    For each completion the reasoning is re-emitted as tagged steps; the PRM's
    probability of the "+" token at every step tag is read off the logits, and the
    completion's score is the *minimum* step probability - a chain of reasoning is
    only as good as its weakest step, which is the standard Math-Shepherd aggregation.

    Args:
        unload_after: free the PRM from VRAM once the group is scored. Needed when the
            policy, the PRM and G=8 generations cannot co-exist in memory; costs a full
            model reload per step, so leave it off whenever VRAM allows.
    """
    if not completions:
        return []

    if not _load_prm_model(model_id):
        return [0.5] * len(completions)

    rewards = []
    with torch.no_grad():
        for comp in completions:
            try:
                text = f"{prompt} {_tag_steps_for_prm(comp)}"
                input_ids = _prm_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=1024
                ).input_ids.to(_prm_model.device)

                logits = _prm_model(input_ids).logits[:, :, _prm_candidate_token_ids]
                scores = logits.softmax(dim=-1)[:, :, 0]  # P(good) at every position
                step_scores = scores[input_ids == _prm_step_tag_id]

                if step_scores.numel() == 0:
                    rewards.append(0.5)
                else:
                    rewards.append(float(step_scores.min().item()))
            except Exception as e:
                print(f"[Rewards] PRM scoring failed for one completion ({e}); using 0.5.")
                rewards.append(0.5)

    if unload_after:
        unload_prm_model()

    return rewards


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r'-?\d[\d,]*\.?\d*')


def extract_final_answer(completion: str) -> str:
    """Pulls the answer out of the `Final Answer: <val>` line the SFT template teaches."""
    match = re.search(r'Final Answer:\s*(.*)', completion, re.IGNORECASE)
    if not match:
        return ""
    # Only the first line after the marker - the model sometimes keeps rambling.
    return match.group(1).strip().split('\n')[0].strip()


def normalize_answer(ans: str) -> str:
    """
    Canonicalises an answer for comparison.

    Handles the three answer shapes in this pipeline: GSM8K numbers (strip $ , %
    and trailing zeros so `1,200` == `$1200.00`), StrategyQA yes/no, and AQuA/MMLU
    option letters.
    """
    if ans is None:
        return ""
    s = str(ans).strip().strip('.').strip()
    if not s:
        return ""

    low = s.lower()
    if low in ("yes", "true"):
        return "yes"
    if low in ("no", "false"):
        return "no"

    # Bare option letter, e.g. "C" or "(C)" or "C)"
    letter = re.fullmatch(r'\(?\s*([A-Ea-e])\s*\)?', s)
    if letter:
        return letter.group(1).upper()

    cleaned = s.replace('$', '').replace('%', '').replace(' ', '')
    nums = _NUM_RE.findall(cleaned)
    if nums:
        try:
            val = float(nums[0].replace(',', ''))
            # Render integers without a trailing ".0" so 18 == 18.0
            if val == int(val):
                return str(int(val))
            return f"{val:g}"
        except ValueError:
            pass

    return low


def answers_match(predicted: str, ground_truth: str) -> bool:
    norm_gt = normalize_answer(ground_truth)
    if norm_gt == "":
        return False
    return normalize_answer(predicted) == norm_gt


# ---------------------------------------------------------------------------
# The six components
# ---------------------------------------------------------------------------

def _outcome_reward(completions: list[str], ground_truth: str) -> list[float]:
    """Verifiable final-answer match, normalised for formatting differences."""
    return [1.0 if answers_match(extract_final_answer(c), ground_truth) else 0.0
            for c in completions]


def _consensus_reward(completions: list[str], group_context: dict) -> list[float]:
    """
    Wrapper around the ICR layer.

    The clustering happens in `icr.py` and the trainer passes the pre-computed
    per-completion rewards through `group_context`, so nothing is re-clustered here.
    0.5 is the neutral value (see `icr.compute_consensus_reward`).
    """
    icr = (group_context or {}).get("icr_rewards")
    if icr is None:
        return [0.5] * len(completions)
    icr = list(icr)
    if len(icr) < len(completions):
        icr = icr + [0.5] * (len(completions) - len(icr))
    return [float(x) for x in icr[:len(completions)]]


def _consistency_reward(completions: list[str]) -> list[float]:
    """
    Lightweight heuristic to check if the final answer logically follows from the
    stated reasoning: the answer value should appear in the tail of the CoT.
    """
    rewards = []
    for comp in completions:
        match = re.search(r'Final Answer:\s*(.*)', comp, re.IGNORECASE)
        if not match:
            rewards.append(0.0)
            continue

        ans = normalize_answer(match.group(1).strip().split('\n')[0])
        cot = comp[:match.start()].strip()
        if not ans or not cot:
            rewards.append(0.0)
            continue

        # Look at the last 200 characters of CoT, comparing normalised numbers so
        # "1,200" in the reasoning matches a "1200" answer.
        tail = cot[-200:]
        tail_values = {normalize_answer(n) for n in _NUM_RE.findall(tail)}
        if ans in tail_values or ans in tail.lower():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def _self_correction_reward(completions: list[str]) -> list[float]:
    """Rewards presence of structured self-correction markers."""
    markers = [
        r"wait,?\s+let me reconsider",
        r"let me (double[- ])?check",
        r"on second thought",
        r"actually,?\s+that'?s (incorrect|wrong)",
        r"let me re-?evaluate",
        r"let me verify",
        r"but wait",
        r"that can'?t be right",
    ]
    rewards = []
    for comp in completions:
        score = 0.0
        for marker in markers:
            if re.search(marker, comp, re.IGNORECASE):
                score = 1.0
                break
        rewards.append(score)
    return rewards


def _format_reward(completions: list[str]) -> list[float]:
    """
    Rewards adherence to the expected CoT structure.

    Scored on the *completion* only - "Let's think step by step." lives in the prompt,
    so checking for it here would make the component unearnable. Capped at 0.05 of the
    total reward on purpose: weighting format any higher reliably produces reward
    hacking, where the model emits perfect scaffolding around degrading reasoning.
    """
    rewards = []
    for comp in completions:
        score = 0.0
        # Some visible reasoning before the answer
        body = re.split(r'Final Answer:', comp, flags=re.IGNORECASE)[0]
        if len(body.strip()) >= 20:
            score += 0.5
        # Exactly one final-answer marker, and it is non-empty
        markers = re.findall(r'Final Answer:', comp, re.IGNORECASE)
        if len(markers) == 1 and extract_final_answer(comp):
            score += 0.5
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "outcome": 0.40,
    "prm": 0.20,
    "consensus": 0.20,
    "consistency": 0.10,
    "self_correction": 0.05,
    "format": 0.05,
}


def resolve_weights(config: dict) -> dict:
    """
    Reads reward weights from config and redistributes the weight of any disabled
    component (`use_prm: false` / `use_icr: false`) proportionally over the rest, so
    the reward stays on a [0, 1] scale across ablations and runs stay comparable.
    """
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in (config.get("reward_weights") or {}).items()
                    if k in DEFAULT_WEIGHTS})

    disabled = []
    if not config.get("use_prm", True):
        disabled.append("prm")
    if not config.get("use_icr", True):
        disabled.append("consensus")

    if not disabled:
        return weights

    if config.get("redistribute_disabled_reward_weight", True):
        freed = sum(weights[d] for d in disabled)
        for d in disabled:
            weights[d] = 0.0
        remaining = sum(weights.values())
        if remaining > 0:
            for k in weights:
                weights[k] += freed * (weights[k] / remaining)
    else:
        for d in disabled:
            weights[d] = 0.0

    return weights


def compute_total_reward(prompt: str, completions: list[str], ground_truth: str,
                         group_context: dict, config: dict) -> tuple[list[float], dict]:
    """
    Combines the 6 reward components based on config weights.

    Returns:
        tuple: (list of total rewards, dict of per-component averages for logging)
    """
    config = config or {}
    group_context = group_context or {}
    weights = resolve_weights(config)

    r_outcome = _outcome_reward(completions, ground_truth)

    if config.get("use_prm", True) and weights["prm"] > 0:
        r_prm = _prm_reward(
            prompt, completions,
            unload_after=config.get("prm_unload_after_scoring", False),
            model_id=config.get("prm_model_id", PRM_MODEL_ID),
        )
    else:
        r_prm = [0.0] * len(completions)

    r_consensus = _consensus_reward(completions, group_context)
    r_consistency = _consistency_reward(completions)
    r_self_corr = _self_correction_reward(completions)
    r_format = _format_reward(completions)

    total_rewards = []
    for i in range(len(completions)):
        tot = (
            weights["outcome"] * r_outcome[i] +
            weights["prm"] * r_prm[i] +
            weights["consensus"] * r_consensus[i] +
            weights["consistency"] * r_consistency[i] +
            weights["self_correction"] * r_self_corr[i] +
            weights["format"] * r_format[i]
        )
        total_rewards.append(float(tot))

    separated_averages = {
        "outcome": float(np.mean(r_outcome)) if completions else 0.0,
        "prm": float(np.mean(r_prm)) if completions else 0.0,
        "consensus": float(np.mean(r_consensus)) if completions else 0.0,
        "consistency": float(np.mean(r_consistency)) if completions else 0.0,
        "self_correction": float(np.mean(r_self_corr)) if completions else 0.0,
        "format": float(np.mean(r_format)) if completions else 0.0,
        "total": float(np.mean(total_rewards)) if completions else 0.0,
    }

    return total_rewards, separated_averages
