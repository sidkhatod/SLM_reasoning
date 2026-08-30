"""GSM8K evaluation (1,319 test problems). Greedy pass@1 + sampled pass@k."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.harness import add_common_args, load_model, run_eval
from utils.data import load_reasoning_dataset


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate on GSM8K"))
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    examples = load_reasoning_dataset(names=["gsm8k"], split="test")
    run_eval(args, model, tokenizer, examples, "gsm8k")


if __name__ == "__main__":
    main()
