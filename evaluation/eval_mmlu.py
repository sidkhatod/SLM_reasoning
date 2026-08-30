"""
MMLU evaluation (bonus benchmark, not a primary KPI).

MMLU is roughly 80% static knowledge recall, so an RL pipeline that targets dynamic
reasoning is expected to move it only modestly. It is measured mainly as a regression
guard: reasoning gains should not come at the cost of general knowledge.

The full test set is 14k questions; a seeded 1,000-question sample is used by default
so a run is affordable. Pass --limit 0 to evaluate everything.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.harness import add_common_args, load_model, run_eval
from utils.data import build_prompt


def load_mmlu(sample_size=1000, seed=42):
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    examples = []
    for ex in dataset:
        choices = ex["choices"]
        choices_text = "\n".join(f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))
        question = f"{ex['question']}\nChoices:\n{choices_text}"
        examples.append({
            "question": question,
            "prompt": build_prompt(question),
            # MMLU stores the answer as an integer index.
            "answer": chr(ord('A') + int(ex["answer"])),
            "source": "mmlu",
        })
    return examples


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate on MMLU"))
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Seeded subsample of the test set (0 = full 14k set)")
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    examples = load_mmlu(sample_size=args.sample_size or None)
    print(f"[data] mmlu (test): {len(examples)} examples")
    run_eval(args, model, tokenizer, examples, "mmlu")


if __name__ == "__main__":
    main()
