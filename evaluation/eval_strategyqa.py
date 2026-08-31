"""StrategyQA evaluation (490 test questions, verifiable yes/no)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.harness import add_common_args, load_model, run_eval
from utils.data import load_reasoning_dataset


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate on StrategyQA"))
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    examples = load_reasoning_dataset(names=["strategyqa"], split="test")
    run_eval(args, model, tokenizer, examples, "strategyqa")


if __name__ == "__main__":
    main()
