import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.layers.dsac import (
    compute_prefix_validity,
    compute_semantic_uncertainty,
    score_problem,
    CurriculumBuffer
)

def test():
    print("Loading test model (gpt2)...")
    # Using tiny gpt2 to make the test run fast and not require a 16GB GPU
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.to("cpu")
    except Exception as e:
        print(f"Failed to load gpt2: {e}")
        return
        
    # Toy dataset of varying difficulties
    # Easy: basic greeting
    # Medium: simple math
    # Hard: complex physics
    
    toy_dataset = [
        {"prompt": "Hello! How are"},
        {"prompt": "Question: What is 2 + 2? \nAnswer: Let's think step by step."},
        {"prompt": "Question: Explain the derivation of the Yang-Mills mass gap using non-perturbative string theory. \nAnswer: Let's think step by step."}
    ]
    
    config = {
        "dsac_target_validity": 0.5,
        "dsac_target_uncertainty": 0.5,
        "dsac_validity_weight": 1.0,
        "dsac_uncertainty_weight": 1.0
    }
    
    print("\n--- Test 1: compute_prefix_validity ---")
    validities = []
    for item in toy_dataset:
        prompt = item["prompt"]
        # Fake a partial completion
        partial = " The answer is"
        v = compute_prefix_validity(model, tokenizer, prompt, [partial])
        validities.append(v)
        print(f"Prompt: {prompt[:30]}... | Validity: {v:.4f}")
        
    print("\n--- Test 2: compute_semantic_uncertainty ---")
    uncertainties = []
    for item in toy_dataset:
        prompt = item["prompt"]
        u = compute_semantic_uncertainty(model, tokenizer, prompt, num_samples=3)
        uncertainties.append(u)
        print(f"Prompt: {prompt[:30]}... | Uncertainty: {u:.4f}")
        
    print("\n--- Test 3: score_problem ---")
    scores = []
    for item in toy_dataset:
        prompt = item["prompt"]
        s = score_problem(model, tokenizer, prompt, config)
        scores.append(s)
        print(f"Prompt: {prompt[:30]}... | Productive Score: {s:.4f}")
        
    print("\n--- Test 4: CurriculumBuffer ---")
    buffer = CurriculumBuffer(config)
    # Refresh with a tiny sample
    buffer.refresh(model, tokenizer, toy_dataset, buffer_size=2, sample_size=3)
    
    print("\nBuffer contents (top 2):")
    for b in buffer.buffer:
        print(f"- {b['prompt'][:50]}...")
        
    print("\n--- Test 5: Metrics Tracking ---")
    # Simulate sampling (6 from buffer, 4 from random for a rough 60% ratio)
    from utils.metrics import log_curriculum_metrics, _tracker
    _tracker.buffer_selected_count = 60
    _tracker.random_selected_count = 40
        
    ratio = log_curriculum_metrics(step=50)
    print(f"Logged Productive Zone Ratio at step 50: {ratio:.2f}")

if __name__ == "__main__":
    test()
