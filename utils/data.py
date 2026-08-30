"""
Shared dataset pipeline for both training phases.

Three open-access sources are aggregated:

    GSM8K       7,473 train / 1,319 test   multi-step math, verifiable numeric reward
    AQuA-RAT    ~3,000 train /   254 test   algebraic reasoning with rationales
    StrategyQA  2,290 train /   490 test   implicit multi-hop commonsense, yes/no reward

Both phases share one prompt template so the format the SFT warmup teaches is exactly
the format the GRPO format/consistency rewards score, and the format the evaluation
harness parses.
"""

import re

PROMPT_TEMPLATE = "Question: {question}\n\nAnswer: Let's think step by step.\n"
COMPLETION_TEMPLATE = "{cot}\n\nFinal Answer: {answer}"

# HF hub ids, tried in order - the canonical namespaced id first, then legacy aliases.
DATASET_SOURCES = {
    "gsm8k": ["openai/gsm8k", "gsm8k"],
    "aqua_rat": ["deepmind/aqua_rat", "aqua_rat"],
    "strategyqa": ["ChilleD/StrategyQA", "wics/strategy-qa", "tau/strategyqa"],
}

DATASET_CONFIGS = {
    "gsm8k": "main",
    "aqua_rat": "raw",
    "strategyqa": None,
}


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def build_completion(cot: str, answer: str) -> str:
    return COMPLETION_TEMPLATE.format(cot=cot.strip(), answer=str(answer).strip())


def _load(name: str, split: str):
    """Loads a dataset by logical name, trying each known hub id in turn."""
    from datasets import load_dataset

    last_err = None
    for hub_id in DATASET_SOURCES[name]:
        try:
            cfg = DATASET_CONFIGS[name]
            if cfg:
                return load_dataset(hub_id, cfg, split=split)
            return load_dataset(hub_id, split=split)
        except Exception as e:  # dataset moved, gated, or split missing
            last_err = e
    raise RuntimeError(f"Could not load '{name}' ({split}): {last_err}")


# ---------------------------------------------------------------------------
# Per-dataset normalisation into {question, cot, answer, source}
# ---------------------------------------------------------------------------

def _norm_gsm8k(x):
    parts = x["answer"].split("####")
    return {
        "question": x["question"],
        "cot": re.sub(r'<<[^>]*>>', '', parts[0]).strip(),  # drop calculator annotations
        "answer": parts[1].strip().replace(',', '') if len(parts) > 1 else "",
        "source": "gsm8k",
    }


def _norm_aqua(x):
    options = "\n".join(x["options"]) if isinstance(x["options"], list) else str(x["options"])
    return {
        "question": f"{x['question']}\nOptions:\n{options}",
        "cot": (x.get("rationale") or "").strip(),
        "answer": str(x.get("correct", "")).strip(),
        "source": "aqua_rat",
    }


def _norm_strategyqa(x):
    question = x.get("question") or x.get("input") or ""
    answer = x.get("answer", x.get("target", ""))
    if isinstance(answer, bool):
        answer = "yes" if answer else "no"
    else:
        answer = str(answer).strip().lower()
        if answer in ("true", "1"):
            answer = "yes"
        elif answer in ("false", "0"):
            answer = "no"

    facts = x.get("facts") or x.get("evidence") or []
    if isinstance(facts, list) and facts and isinstance(facts[0], str):
        cot = "\n".join(f"- {f}" for f in facts)
    else:
        cot = str(x.get("explanation") or "").strip()

    return {"question": question, "cot": cot, "answer": answer, "source": "strategyqa"}


_NORMALIZERS = {
    "gsm8k": _norm_gsm8k,
    "aqua_rat": _norm_aqua,
    "strategyqa": _norm_strategyqa,
}

# Reported test-split sizes, used to keep the eval harness honest about what it ran on.
TEST_SPLIT_SIZES = {"gsm8k": 1319, "aqua_rat": 254, "strategyqa": 490}


def load_reasoning_dataset(names=("gsm8k", "aqua_rat", "strategyqa"), split: str = "train",
                           limits: dict = None, seed: int = 42, verbose: bool = True) -> list[dict]:
    """
    Loads and normalises the requested datasets into a flat list of dicts with keys
    `question`, `cot`, `answer`, `source`, plus the ready-to-use `prompt`.

    Args:
        names: which datasets to include.
        split: "train" or "test".
        limits: optional {name: max_examples}. AQuA-RAT defaults to 3,000 train
            examples - the full 97k train split would swamp the other two sources.
        verbose: print a per-source summary.

    A source that fails to load is skipped with a warning rather than killing the run;
    if none load, a RuntimeError is raised.
    """
    limits = dict(limits or {})
    if split == "train":
        limits.setdefault("aqua_rat", 3000)

    combined = []
    for name in names:
        try:
            raw = _load(name, split)
        except Exception as e:
            print(f"[data] Skipping {name} ({split}): {e}")
            continue

        norm = _NORMALIZERS[name]
        items = []
        for ex in raw:
            try:
                item = norm(ex)
            except Exception:
                continue
            if not item["question"] or not item["answer"]:
                continue
            item["prompt"] = build_prompt(item["question"])
            items.append(item)

        cap = limits.get(name)
        if cap is not None and len(items) > cap:
            import random as _random
            rng = _random.Random(seed)
            items = rng.sample(items, cap)

        if verbose:
            print(f"[data] {name} ({split}): {len(items)} examples")
        combined.extend(items)

    if not combined:
        raise RuntimeError(
            "No datasets could be loaded. Check the dataset ids, your internet "
            "connection, or your Hugging Face credentials."
        )

    import random as _random
    _random.Random(seed).shuffle(combined)
    if verbose:
        print(f"[data] Combined {split} set: {len(combined)} examples")
    return combined


def train_val_split(dataset: list[dict], val_size: int = 200, seed: int = 42):
    """
    Carves a validation slice off the training data.

    Checkpoint selection runs against this held-out slice rather than the final
    training step, which is the guard against selecting a reward-hacked policy.
    """
    import random as _random

    items = list(dataset)
    _random.Random(seed).shuffle(items)
    val_size = min(val_size, max(0, len(items) // 5))
    return items[val_size:], items[:val_size]
