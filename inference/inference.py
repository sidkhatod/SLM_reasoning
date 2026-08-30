"""
Inference for a trained PRISM-GRPO adapter.

All three layers are training-time only - SPM shapes gradients, ICR reads the GRPO
generation group, DSAC selects training problems - so serving is a plain base model
plus one LoRA adapter, with zero inference-time overhead. That is the deployment
property the whole design is built around, and this script is what demonstrates it.

    # single question
    python inference/inference.py --checkpoint_path outputs/a6_full_prism_grpo/best \
        --question "Natalia sold clips to 48 friends in April, and half as many in May. How many did she sell altogether?"

    # interactive
    python inference/inference.py --checkpoint_path outputs/a6_full_prism_grpo/best --interactive

    # merge the adapter into the base weights for standalone deployment
    python inference/inference.py --checkpoint_path outputs/a6_full_prism_grpo/best \
        --merge_and_save outputs/prism_merged
"""

import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rewards.reward_functions import extract_final_answer
from utils.logger import configure_console
from utils.data import build_prompt


def load_model(base_model, checkpoint_path, quantize=True, merge=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_src = (checkpoint_path if checkpoint_path and os.path.isdir(checkpoint_path)
                     else base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if quantize and torch.cuda.is_available() and not merge:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        # Merging into 4-bit weights is lossy, so a merge run loads in half/full precision.
        load_kwargs["torch_dtype"] = (torch.bfloat16 if torch.cuda.is_available()
                                      else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        if merge:
            print("Merging LoRA adapter into the base weights...")
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=512, temperature=0.0,
             num_samples=1):
    """Runs the SFT/GRPO prompt template and returns (completions, extracted answers)."""
    prompt = build_prompt(question)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    do_sample = temperature > 0
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completions = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
    return completions, [extract_final_answer(c) for c in completions]


def majority_vote(answers):
    """Self-consistency: the most common non-empty answer across sampled decodes."""
    from collections import Counter

    votes = Counter(a for a in answers if a)
    return votes.most_common(1)[0][0] if votes else ""


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="PRISM-GRPO inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="LoRA adapter directory (omit for the raw base model)")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy; > 0 samples")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="With temperature > 0, majority-vote over N samples")
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument("--merge_and_save", type=str, default=None,
                        help="Merge the adapter into the base model and save it here")
    args = parser.parse_args()

    if not args.question and not args.interactive and not args.merge_and_save:
        parser.error("Pass --question, --interactive, or --merge_and_save.")

    model, tokenizer = load_model(
        args.base_model, args.checkpoint_path,
        quantize=not args.no_quantization,
        merge=bool(args.merge_and_save),
    )

    if args.merge_and_save:
        os.makedirs(args.merge_and_save, exist_ok=True)
        model.save_pretrained(args.merge_and_save)
        tokenizer.save_pretrained(args.merge_and_save)
        print(f"Merged model saved to {args.merge_and_save}")
        if not args.question and not args.interactive:
            return

    def answer(question):
        t0 = time.time()
        completions, answers = generate(
            model, tokenizer, question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples,
        )
        print("\n" + "-" * 60)
        print(completions[0].strip())
        print("-" * 60)
        final = majority_vote(answers) if len(answers) > 1 else answers[0]
        if len(answers) > 1:
            print(f"Sampled answers: {answers}")
        print(f"Answer: {final or '(none extracted)'}   [{time.time() - t0:.1f}s]\n")

    if args.question:
        answer(args.question)

    if args.interactive:
        print("Interactive mode - type a question, or 'quit' to exit.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            answer(q)


if __name__ == "__main__":
    main()
