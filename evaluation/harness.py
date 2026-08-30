"""
Shared evaluation harness for GSM8K / StrategyQA / MMLU.

pass@1 is scored on a *greedy* decode and pass@k on k sampled decodes. The original
scripts sampled at temperature 0.7 and called the first sample "pass@1", which is a
noisier and systematically lower number than the greedy pass@1 everyone reports.

Answer extraction and normalisation are imported from `rewards.reward_functions`, so
the string a completion has to produce to earn the training outcome reward is exactly
the string that counts as correct at evaluation time.
"""

import json
import os
import sys

import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rewards.reward_functions import answers_match, extract_final_answer  # noqa: E402
from utils.data import build_prompt  # noqa: E402


def add_common_args(parser):
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="LoRA adapter directory. Omit to evaluate the raw base model.")
    parser.add_argument("--k", type=int, default=4, help="Pass@k sampled completions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (smoke tests / budget)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_name", type=str, default=None,
                        help="Override the result filename stem")
    parser.add_argument("--no_quantization", action="store_true")
    return parser


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_src = (args.checkpoint_path
                     if args.checkpoint_path and os.path.isdir(args.checkpoint_path)
                     else args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if torch.cuda.is_available() and not args.no_quantization:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32 if not torch.cuda.is_available() else torch.bfloat16

    print(f"Loading {args.base_model}"
          + (f" + adapter {args.checkpoint_path}" if args.checkpoint_path else " (base, no adapter)")
          + "...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    if args.checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path)

    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt, max_new_tokens, num_return_sequences,
              do_sample, temperature):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_eval(args, model, tokenizer, examples, task_name, match_fn=None):
    """
    Args:
        examples: list of {"question": str, "answer": str} (already prompt-ready text).
        match_fn: optional custom comparison; defaults to the training-time matcher.
    """
    match_fn = match_fn or (lambda pred, gt: answers_match(pred, gt))

    if args.limit:
        examples = examples[:args.limit]

    results = []
    correct_at_1 = 0
    correct_at_k = 0

    print(f"Running {task_name} on {len(examples)} examples "
          f"(greedy pass@1 + {args.k} sampled for pass@{args.k})...")

    with torch.no_grad():
        for ex in tqdm(examples):
            prompt = ex.get("prompt") or build_prompt(ex["question"])
            gt = ex["answer"]

            greedy = _generate(model, tokenizer, prompt, args.max_new_tokens, 1, False, None)[0]
            greedy_answer = extract_final_answer(greedy)
            pass_1 = bool(match_fn(greedy_answer, gt))

            sampled, sampled_answers = [], []
            if args.k > 1:
                sampled = _generate(model, tokenizer, prompt, args.max_new_tokens,
                                    args.k, True, args.temperature)
                sampled_answers = [extract_final_answer(c) for c in sampled]

            # pass@k includes the greedy decode, matching how the checkpoint is served.
            pass_k = pass_1 or any(match_fn(a, gt) for a in sampled_answers)

            correct_at_1 += int(pass_1)
            correct_at_k += int(pass_k)

            results.append({
                "question": ex["question"],
                "ground_truth": gt,
                "greedy_completion": greedy,
                "greedy_answer": greedy_answer,
                "sampled_answers": sampled_answers,
                "pass_1": pass_1,
                "pass_k": bool(pass_k),
            })

    total = max(1, len(examples))
    metrics = {
        "pass@1": correct_at_1 / total,
        f"pass@{args.k}": correct_at_k / total,
        "n_examples": len(examples),
    }

    os.makedirs("outputs/eval_results", exist_ok=True)
    stem = args.output_name or os.path.basename(
        os.path.normpath(args.checkpoint_path or args.base_model)
    )
    out_file = f"outputs/eval_results/{stem}__{task_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "checkpoint": args.checkpoint_path,
                   "base_model": args.base_model, "results": results}, f, indent=2)

    print("\n" + "=" * 46)
    print(f"{task_name.upper()} Evaluation Summary")
    print("=" * 46)
    print(f"Model:            {stem}")
    print(f"Examples:         {len(examples)}")
    print(f"Pass@1 (greedy):  {metrics['pass@1'] * 100:.2f}%")
    print(f"Pass@{args.k} (sampled): {metrics[f'pass@{args.k}'] * 100:.2f}%")
    print(f"Details:          {out_file}")
    print("=" * 46 + "\n")

    return metrics
