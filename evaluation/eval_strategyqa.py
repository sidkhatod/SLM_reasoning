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
    """Extracts the final yes/no answer from StrategyQA format."""
    match = re.search(r'Final Answer:\s*(yes|no)', completion, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    # Fallback heuristic
    completion_lower = completion.lower()
    if completion_lower.endswith("yes") or completion_lower.endswith("yes."):
        return "yes"
    if completion_lower.endswith("no") or completion_lower.endswith("no."):
        return "no"
        
    return "unknown"

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

    print("Loading StrategyQA test set...")
    # There are multiple StrategyQA variants on HF. Using a popular one that has a validation/test split.
    # Alternatively, we can use the original dataset format.
    try:
        dataset = load_dataset("wics/strategy-qa", split="test")
    except:
        try:
            # Fallback if first doesn't work in this specific environment
            dataset = load_dataset("ChilleD/StrategyQA", split="test")
        except:
            # Another common one
            dataset = load_dataset("tau/strategy_qa", split="train")
            # Just take the last 490 examples as "test" if there's no test split
            dataset = dataset.select(range(len(dataset)-490, len(dataset)))
            
    # Subsample if dataset is too large, the prompt mentioned 490 test examples
    if len(dataset) > 490:
        dataset = dataset.select(range(490))
    
    results = []
    correct_at_1 = 0
    correct_at_k = 0
    
    print(f"Running inference (k={args.k})...")
    with torch.no_grad():
        for example in tqdm(dataset):
            q_text = example.get('question', example.get('input', ''))
            a_val = example.get('answer', example.get('target', ''))
            
            # Map boolean to string if necessary
            if isinstance(a_val, bool):
                gt_ans = "yes" if a_val else "no"
            else:
                gt_ans = str(a_val).lower().strip()
                
            prompt = f"Question: {q_text}\nAnswer: Let's think step by step."
            
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
            
            # Check correctness
            is_correct = [ans == gt_ans for ans in extracted_answers]
            
            pass_1 = is_correct[0] if len(is_correct) > 0 else False
            pass_k = any(is_correct)
            
            if pass_1:
                correct_at_1 += 1
            if pass_k:
                correct_at_k += 1
                
            results.append({
                "question": q_text,
                "ground_truth": gt_ans,
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
    out_file = f"outputs/eval_results/{chkpt_name}_strategyqa.json"
    with open(out_file, "w") as f:
        json.dump({
            "metrics": {"pass@1": acc_1, f"pass@{args.k}": acc_k},
            "results": results
        }, f, indent=2)

    print("\n" + "="*40)
    print("StrategyQA Evaluation Summary")
    print("="*40)
    print(f"Model: {chkpt_name}")
    print(f"Total Examples: {total}")
    print(f"Pass@1 Accuracy: {acc_1*100:.2f}%")
    print(f"Pass@{args.k} Accuracy: {acc_k*100:.2f}%")
    print(f"Saved detailed results to {out_file}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
