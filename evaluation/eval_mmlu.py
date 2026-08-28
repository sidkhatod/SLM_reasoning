import argparse
import json
import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def extract_answer(completion: str):
    """Extracts the final multiple-choice answer from MMLU format."""
    match = re.search(r'Final Answer:\s*([A-D])', completion, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback heuristic
    completion_upper = completion.upper()
    for letter in ['A', 'B', 'C', 'D']:
        if completion_upper.endswith(letter) or completion_upper.endswith(f"{letter}."):
            return letter
    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=4, help="Pass@k completions to generate")
    args = parser.parse_args()

    print(f"Loading {args.base_model} + {args.checkpoint_path} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    model.eval()

    print("Loading MMLU test set...")
    # Loading a standard subset to keep evaluation reasonable in time, e.g. "all" or specific subjects.
    # To keep it bounded for this script, we take a balanced sample from "cais/mmlu" "all" config.
    try:
        dataset = load_dataset("cais/mmlu", "all", split="test")
        # Subsample to ~1000 items to match GSM8K scale for quick evals
        if len(dataset) > 1000:
            dataset = dataset.shuffle(seed=42).select(range(1000))
    except:
        print("Could not load cais/mmlu, falling back to a tiny dummy for testing.")
        dataset = [{"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}]

    results = []
    correct_at_1 = 0
    correct_at_k = 0
    
    print(f"Running inference (k={args.k})...")
    with torch.no_grad():
        for example in tqdm(dataset):
            q = example['question']
            choices = example['choices']
            
            # MMLU answer is an integer index
            ans_idx = example['answer']
            gt_letter = chr(ord('A') + ans_idx)
            
            choices_text = "\n".join([f"{chr(ord('A')+i)}) {c}" for i, c in enumerate(choices)])
            prompt = f"Question: {q}\nChoices:\n{choices_text}\nAnswer: Let's think step by step."
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=args.k,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            completions = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            extracted_answers = [extract_answer(c) for c in completions]
            
            is_correct = [ans == gt_letter for ans in extracted_answers]
            
            pass_1 = is_correct[0] if len(is_correct) > 0 else False
            pass_k = any(is_correct)
            
            if pass_1:
                correct_at_1 += 1
            if pass_k:
                correct_at_k += 1
                
            results.append({
                "question": q,
                "ground_truth": gt_letter,
                "completions": completions,
                "extracted_answers": extracted_answers,
                "pass_1": bool(pass_1),
                "pass_k": bool(pass_k)
            })

    total = len(dataset)
    acc_1 = correct_at_1 / total if total > 0 else 0
    acc_k = correct_at_k / total if total > 0 else 0
    
    os.makedirs("outputs/eval_results", exist_ok=True)
    chkpt_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    out_file = f"outputs/eval_results/{chkpt_name}_mmlu.json"
    with open(out_file, "w") as f:
        json.dump({
            "metrics": {"pass@1": acc_1, f"pass@{args.k}": acc_k},
            "results": results
        }, f, indent=2)

    print("\n" + "="*40)
    print("MMLU Evaluation Summary")
    print("="*40)
    print(f"Model: {chkpt_name}")
    print(f"Total Examples: {total}")
    print(f"Pass@1 Accuracy: {acc_1*100:.2f}%")
    print(f"Pass@{args.k} Accuracy: {acc_k*100:.2f}%")
    print(f"Saved detailed results to {out_file}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
