import torch
import numpy as np
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Reuse the frozen MiniLM embedder from the ICR layer
from training.layers.icr import get_embedder
from utils.metrics import record_curriculum_selection

def compute_prefix_validity(model, tokenizer, prompt: str, partial_completions: list[str]) -> float:
    """
    Estimates how 'on track' the early reasoning tokens are.
    Computes the average normalized log probability (validity) of the partial completions.
    Lower probability roughly means the model is already lost early.
    
    Args:
        model: CausalLM model
        tokenizer: Model tokenizer
        prompt: The input prompt
        partial_completions: List of early reasoning step strings
    
    Returns:
        float: Average prefix validity score in [0.0, 1.0]
    """
    if not partial_completions:
        return 0.0
        
    model.eval()
    validities = []
    
    with torch.no_grad():
        for partial in partial_completions:
            text = prompt + partial
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits # (1, seq_len, vocab_size)
            
            input_ids = inputs.input_ids[0]
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            
            # If the prompt is somehow longer than the text, fail gracefully
            if prompt_len >= len(input_ids):
                validities.append(0.0)
                continue
                
            # Focus on the log probabilities of the *partial completion* tokens
            shift_logits = logits[0, prompt_len-1:-1, :].contiguous()
            shift_labels = input_ids[prompt_len:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits, shift_labels)
            
            # Map CrossEntropy (NLL) to a [0,1] probability scale
            prob = torch.exp(-loss).item()
            validities.append(prob)
            
    return float(np.mean(validities))

def compute_semantic_uncertainty(model, tokenizer, prompt: str, num_samples: int = 4) -> float:
    """
    Samples short continuations and estimates uncertainty via semantic divergence.
    """
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30, # Generate short semantic trajectories
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        
    prompt_len = inputs.input_ids.shape[1]
    gen_tokens = outputs[:, prompt_len:]
    samples = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    
    # Embed using frozen MiniLM
    embedder = get_embedder()
    embeddings = embedder.encode(samples, convert_to_tensor=True, show_progress_bar=False)
    
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(embeddings, embeddings.transpose(0, 1))
    
    n = embeddings.size(0)
    if n <= 1:
        return 0.0
        
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    avg_sim = sim_matrix[mask].mean().item()
    
    # Uncertainty is inverse to similarity. 
    # High similarity -> Low Uncertainty.
    uncertainty = 1.0 - avg_sim
    return max(0.0, min(1.0, uncertainty))


def score_problem(model, tokenizer, prompt: str, config: dict) -> float:
    """
    Combines prefix validity and semantic uncertainty into a 'productive zone' score.
    High scores indicate the problem is in the Goldilocks zone (not too easy, not too hard).
    """
    # Generate a single partial prefix for validity check
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        partial = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
    validity = compute_prefix_validity(model, tokenizer, prompt, [partial])
    uncertainty = compute_semantic_uncertainty(model, tokenizer, prompt, num_samples=4)
    
    target_validity = config.get("dsac_target_validity", 0.5)
    target_uncertainty = config.get("dsac_target_uncertainty", 0.5)
    weight_v = config.get("dsac_validity_weight", 1.0)
    weight_u = config.get("dsac_uncertainty_weight", 1.0)
    
    # Measure distance from ideal middle band. 
    # The closer to target, the higher the score.
    dist_v = abs(validity - target_validity)
    dist_u = abs(uncertainty - target_uncertainty)
    
    # Normalized score: peaks at 1.0 when validity/uncertainty are exactly at targets
    score = 1.0 - ((weight_v * dist_v) + (weight_u * dist_u)) / (weight_v + weight_u)
    return max(0.0, score)


class CurriculumBuffer:
    def __init__(self, config: dict):
        self.config = config
        self.buffer = []
        
    def refresh(self, model, tokenizer, full_dataset: list[dict], buffer_size: int, sample_size: int = 100):
        """
        Re-scores a sample of the dataset and updates the buffer with the top problems.
        Should be called every N steps (e.g., 50) to avoid continuous compute cost.
        """
        print(f"\n[DSAC] Refreshing Curriculum Buffer (evaluating {sample_size} problems)...")
        candidates = random.sample(full_dataset, min(sample_size, len(full_dataset)))
        
        scored_candidates = []
        for item in candidates:
            # Flexible extraction depending on dataset format
            if isinstance(item, dict) and "prompt" in item:
                p_str = item["prompt"]
            elif isinstance(item, dict) and "text" in item:
                # Naive extract if SFT text string is present
                p_str = item["text"].split("Answer:")[0] + "Answer:"
            else:
                p_str = str(item)
                
            score = score_problem(model, tokenizer, p_str, self.config)
            scored_candidates.append((score, item))
            
        # Sort descending by productive zone score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take the top N (those closest to the Goldilocks zone)
        self.buffer = [item for score, item in scored_candidates[:buffer_size]]
        
        top_score = scored_candidates[0][0] if scored_candidates else 0
        bot_score = scored_candidates[buffer_size-1][0] if buffer_size <= len(scored_candidates) else 0
        print(f"[DSAC] Buffer refreshed. Top score: {top_score:.3f}, Bottom score in buffer: {bot_score:.3f}")
        
    def sample_problem(self, full_dataset: list, p_buffer: float = 0.5):
        """
        Samples a problem either from the productive buffer or randomly from the full dataset.
        Records the selection for metric tracking.
        """
        if self.buffer and random.random() < p_buffer:
            record_curriculum_selection(from_buffer=True)
            return random.choice(self.buffer)
        else:
            record_curriculum_selection(from_buffer=False)
            return random.choice(full_dataset)
