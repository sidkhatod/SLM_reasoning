import sys
import os
import torch
from transformers import AutoTokenizer

# Ensure we can import layers module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.spm import (
    find_shared_prefix_tokens,
    compute_soft_mask,
    apply_nthr_bonus,
    apply_spm
)

def test():
    # Use a dummy tokenizer for quick testing. GPT2 tokenizer is fast and widely available.
    print("Loading test tokenizer (gpt2)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Failed to load gpt2 tokenizer: {e}. Attempting basic mock tokenizer logic.")
        # If gpt2 isn't installed locally or offline, we mock it
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Simple word-level tokenization for testing
                return text.split()
            def decode(self, token_ids):
                return " ".join(token_ids)
        tokenizer = MockTokenizer()
    
    # Toy example: 
    # Prompt: "What is 2+3*4?"
    # Shared reasoning: "By order of operations, we do multiplication first. 3*4=12. Then 2+12="
    shared_prefix = "By order of operations, we do multiplication first. 3*4=12. Then 2+12="
    
    correct_completions = [
        f"{shared_prefix}14.",  # Correct
    ]
    incorrect_completions = [
        f"{shared_prefix}10.",  # Incorrect (math error at the very end)
        "We do addition first. 2+3=5. 5*4=20." # Incorrect (no shared prefix)
    ]
    
    print("\n--- Test 1: find_shared_prefix_tokens ---")
    shared_pos_list = find_shared_prefix_tokens(correct_completions, incorrect_completions, tokenizer)
    for i, comp in enumerate(incorrect_completions):
        tokens = tokenizer.encode(comp, add_special_tokens=False)
        # Handle string tokens (from mock) vs int tokens (from real tokenizer)
        shared_tokens = []
        for p in sorted(list(shared_pos_list[i])):
            if isinstance(tokens[p], str):
                shared_tokens.append(tokens[p])
            else:
                shared_tokens.append(tokenizer.decode([tokens[p]]))
        
        print(f"\nIncorrect Completion {i+1}: '{comp}'")
        print(f"  -> Shared positions count: {len(shared_pos_list[i])}")
        print(f"  -> Shared tokens (first 5): {shared_tokens[:5]}...")
        
    print("\n--- Test 2: compute_soft_mask ---")
    seq_len = len(tokenizer.encode(incorrect_completions[0], add_special_tokens=False))
    mask = compute_soft_mask(shared_pos_list[0], seq_len, mask_strength=0.5)
    print(f"Mask values for Completion 1 (1.0 = no mask, 0.5 = dampened):")
    # Print the first 10 mask values and the last 3 mask values to show dampening
    print("Start:", mask.tolist()[:10], "... End:", mask.tolist()[-3:])
    
    print("\n--- Test 3: apply_spm (End-to-End) ---")
    group_completions = correct_completions + incorrect_completions
    # Rewards: Correct gets +1.0, Incorrect gets -1.0
    group_rewards = [1.0, -1.0, -1.0]
    
    config = {
        "spm_mask_strength": 0.5,
        "nthr_bonus_strength": 0.1
    }
    
    token_advantages = apply_spm(group_completions, group_rewards, None, tokenizer, config)
    
    for i, adv in enumerate(token_advantages):
        comp = group_completions[i]
        label = "Correct" if group_rewards[i] > 0 else "Incorrect"
        print(f"\n{label} Completion {i+1}:")
        tokens = tokenizer.encode(comp, add_special_tokens=False)
        # Display the advantage of the first few and last few tokens
        print("Advantage pattern:")
        for t_idx in list(range(min(5, len(tokens)))) + list(range(max(0, len(tokens)-3), len(tokens))):
            t_val = tokens[t_idx]
            tok_str = t_val if isinstance(t_val, str) else tokenizer.decode([t_val])
            print(f"  {tok_str:<10} | Adv: {adv[t_idx]:.2f}")

if __name__ == "__main__":
    test()
