import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

# To avoid keeping the PRM in memory forever if not needed, we use a singleton pattern.
_prm_model = None
_prm_tokenizer = None

def _load_prm_model():
    global _prm_model, _prm_tokenizer
    if _prm_model is None:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        print(f"[Rewards] Loading Math-Shepherd PRM ({model_id}) in 4-bit...")
        _prm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        if _prm_tokenizer.pad_token is None:
            _prm_tokenizer.pad_token = _prm_tokenizer.eos_token
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _prm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        _prm_model.eval()

def _unload_prm_model():
    """Unloads the PRM to free up GPU memory."""
    global _prm_model, _prm_tokenizer
    if _prm_model is not None:
        del _prm_model
        del _prm_tokenizer
        _prm_model = None
        _prm_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[Rewards] Unloaded Math-Shepherd PRM from memory.")

def _prm_reward(prompt: str, completions: list[str], sequential: bool = True) -> list[float]:
    """
    Computes step-level quality scores using Math-Shepherd.
    """
    if sequential:
        _load_prm_model()
        
    global _prm_model, _prm_tokenizer
    if _prm_model is None:
        _load_prm_model()

    rewards = []
    # Math-Shepherd evaluates tokens like '+' or '-' after steps. 
    # For a simple and robust integration, we can take the log probability of a correct indicator
    # or just use the model's sequence output. We'll use a placeholder logic for the specific PRM decoding,
    # as the exact token IDs for good/bad depend on the tokenizer.
    
    with torch.no_grad():
        for comp in completions:
            text = prompt + comp
            # In a real Math-Shepherd implementation, we would extract the logits of the specific
            # step-evaluation tokens. For this boilerplate, we'll return a proxy score based on perplexity.
            inputs = _prm_tokenizer(text, return_tensors="pt").to(_prm_model.device)
            try:
                outputs = _prm_model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()
                # Lower loss -> higher reward. Mapped to [0, 1].
                reward = np.exp(-loss)
            except Exception:
                reward = 0.0
            rewards.append(float(reward))

    if sequential:
        _unload_prm_model()

    return rewards

def _outcome_reward(completions: list[str], ground_truth: str) -> list[float]:
    """
    Exact match outcome reward.
    """
    rewards = []
    for comp in completions:
        # Extract the final answer. SFT format usually ends with 'Final Answer: <val>'
        match = re.search(r'Final Answer:\s*(.*)', comp, re.IGNORECASE | re.DOTALL)
        if match:
            ans = match.group(1).strip().strip('.')
            gt = ground_truth.strip().strip('.')
            rewards.append(1.0 if ans == gt else 0.0)
        else:
            rewards.append(0.0)
    return rewards

def _consensus_reward(completions: list[str], group_context: dict) -> list[float]:
    """
    Wrapper around ICR layer.
    """
    # The ICR logic is handled in `icr.py`. We assume the group_context already contains
    # the pre-calculated ICR consensus rewards for each completion to avoid re-clustering.
    # If not, we just return 0s.
    if "icr_rewards" in group_context:
        return group_context["icr_rewards"]
    return [0.0] * len(completions)

def _consistency_reward(completions: list[str]) -> list[float]:
    """
    Lightweight heuristic to check if the final answer logically follows from the stated reasoning.
    Checks if the final answer string is found in the last few sentences of the CoT.
    """
    rewards = []
    for comp in completions:
        match = re.search(r'Final Answer:\s*(.*)', comp, re.IGNORECASE | re.DOTALL)
        if match:
            ans = match.group(1).strip().strip('.')
            cot = comp[:match.start()].strip()
            # Look at the last 100 characters of CoT
            last_part = cot[-100:]
            if ans in last_part:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def _self_correction_reward(completions: list[str]) -> list[float]:
    """
    Rewards presence of structured self-correction markers.
    """
    markers = [
        r"wait, let me reconsider",
        r"let me check",
        r"on second thought",
        r"actually, that's incorrect",
        r"let me re-evaluate"
    ]
    rewards = []
    for comp in completions:
        score = 0.0
        for marker in markers:
            if re.search(marker, comp, re.IGNORECASE):
                score = 1.0
                break
        rewards.append(score)
    return rewards

def _format_reward(completions: list[str]) -> list[float]:
    """
    Rewards adherence to the expected CoT structure.
    """
    rewards = []
    for comp in completions:
        score = 0.0
        if "Let's think step by step." in comp:
            score += 0.5
        if "Final Answer:" in comp:
            score += 0.5
        rewards.append(score)
    return rewards

def compute_total_reward(prompt: str, completions: list[str], ground_truth: str, group_context: dict, config: dict) -> tuple[list[float], dict]:
    """
    Combines the 6 reward components based on config weights.
    
    Returns:
        tuple: (list of total rewards, dict of separated reward component averages for logging)
    """
    weights = config.get("reward_weights", {})
    w_outcome = weights.get("outcome", 0.40)
    w_prm = weights.get("prm", 0.20)
    w_consensus = weights.get("consensus", 0.20)
    w_consistency = weights.get("consistency", 0.10)
    w_self_correction = weights.get("self_correction", 0.05)
    w_format = weights.get("format", 0.05)
    
    sequential_prm = config.get("sequential_prm_scoring", True)
    
    r_outcome = _outcome_reward(completions, ground_truth)
    r_prm = _prm_reward(prompt, completions, sequential=sequential_prm)
    r_consensus = _consensus_reward(completions, group_context)
    r_consistency = _consistency_reward(completions)
    r_self_corr = _self_correction_reward(completions)
    r_format = _format_reward(completions)
    
    total_rewards = []
    for i in range(len(completions)):
        tot = (
            w_outcome * r_outcome[i] +
            w_prm * r_prm[i] +
            w_consensus * r_consensus[i] +
            w_consistency * r_consistency[i] +
            w_self_correction * r_self_corr[i] +
            w_format * r_format[i]
        )
        total_rewards.append(tot)
        
    separated_averages = {
        "outcome": np.mean(r_outcome),
        "prm": np.mean(r_prm),
        "consensus": np.mean(r_consensus),
        "consistency": np.mean(r_consistency),
        "self_correction": np.mean(r_self_corr),
        "format": np.mean(r_format),
        "total": np.mean(total_rewards)
    }
        
    return total_rewards, separated_averages
