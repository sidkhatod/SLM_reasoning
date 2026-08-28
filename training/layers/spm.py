import torch

def find_shared_prefix_tokens(correct_completions: list[str], incorrect_completions: list[str], tokenizer) -> list[set[int]]:
    """
    Identifies token positions where the reasoning content overlaps between a group's
    correct and incorrect completions.
    
    Why this is needed:
    Negative gradient updates on incorrect completions can accidentally suppress tokens 
    that are actually correct reasoning steps, simply because those tokens appear as a 
    shared prefix before the incorrect completion diverges. This causes "Lazy Likelihood 
    Displacement" (LLD), where the model unlearns valid reasoning patterns. 
    SPM detects these shared prefixes to mask them from the negative gradient.
    
    Args:
        correct_completions: List of string completions that are labeled as correct.
        incorrect_completions: List of string completions that are labeled as incorrect.
        tokenizer: The tokenizer used for the model.
        
    Returns:
        A list of sets, one for each incorrect completion. Each set contains the token 
        indices in that incorrect completion that are part of a shared prefix with ANY 
        correct completion.
    """
    shared_positions_per_incorrect = []
    
    # Tokenize completions
    correct_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in correct_completions]
    incorrect_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in incorrect_completions]
    
    for inc_toks in incorrect_tokens:
        shared_pos = set()
        for cor_toks in correct_tokens:
            # Exact token-ID matching as a first pass.
            # In a full implementation, a semantic similarity check (e.g. comparing 
            # token windows via embedding similarity) can be swapped in here to handle 
            # paraphrased-but-equivalent steps, avoiding the brittleness of exact match.
            
            # Simple exact prefix match (finds the longest common prefix)
            max_len = min(len(inc_toks), len(cor_toks))
            for i in range(max_len):
                if inc_toks[i] == cor_toks[i]:
                    shared_pos.add(i)
                else:
                    # Break at the first divergence for strict prefix matching
                    break
        shared_positions_per_incorrect.append(shared_pos)
        
    return shared_positions_per_incorrect

def find_shared_correct_tokens(correct_completions: list[str], incorrect_completions: list[str], tokenizer) -> list[set[int]]:
    """
    Companion to find_shared_prefix_tokens. Identifies which tokens in the CORRECT 
    completions are shared with the incorrect ones, so we can apply the NTHR bonus.
    """
    shared_positions_per_correct = []
    
    correct_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in correct_completions]
    incorrect_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in incorrect_completions]
    
    for cor_toks in correct_tokens:
        shared_pos = set()
        for inc_toks in incorrect_tokens:
            max_len = min(len(inc_toks), len(cor_toks))
            for i in range(max_len):
                if inc_toks[i] == cor_toks[i]:
                    shared_pos.add(i)
                else:
                    break
        shared_positions_per_correct.append(shared_pos)
        
    return shared_positions_per_correct


def compute_soft_mask(shared_token_positions: set[int], sequence_length: int, mask_strength: float = 0.5) -> torch.Tensor:
    """
    Returns a per-token multiplier in [0,1] to apply to the negative gradient at 
    shared prefix positions.
    
    Why soft masking:
    Instead of hard-masking (zeroing out) the gradient on shared tokens, soft masking
    multiplies the negative advantage by a dampening factor (e.g., 0.5). This approach
    preserves some learning signal in case the shared tokens are actually suboptimal,
    while mitigating severe Lazy Likelihood Displacement.
    
    Args:
        shared_token_positions: Set of integer token indices that are shared.
        sequence_length: Total number of tokens in the completion.
        mask_strength: How much to dampen the gradient (0.0 = no masking, 1.0 = full masking).
        
    Returns:
        A 1D tensor of shape (sequence_length,) containing multipliers.
    """
    # Initialize multipliers to 1.0 (no dampening)
    mask = torch.ones(sequence_length, dtype=torch.float32)
    
    # Apply dampening to shared positions
    multiplier = 1.0 - mask_strength
    for pos in shared_token_positions:
        if pos < sequence_length:
            mask[pos] = multiplier
            
    return mask

def apply_nthr_bonus(shared_correct_token_positions: set[int], sequence_length: int, bonus_strength: float = 0.1) -> torch.Tensor:
    """
    The NTHR extension: gives a small POSITIVE signal to tokens that are shared 
    AND come from a correct completion, on top of the masking above.
    
    Why this is needed:
    If a correct completion and an incorrect completion share a valid reasoning prefix,
    the model might still slightly penalize the correct prefix from the incorrect trajectory.
    NTHR counters this by slightly boosting the reward for those specific shared tokens
    in the successful trajectory.
    
    Args:
        shared_correct_token_positions: Set of integer token indices in the correct completion.
        sequence_length: Total number of tokens in the completion.
        bonus_strength: The additive bonus to apply to the advantage.
        
    Returns:
        A 1D tensor of shape (sequence_length,) containing the additive bonuses.
    """
    bonus = torch.zeros(sequence_length, dtype=torch.float32)
    for pos in shared_correct_token_positions:
        if pos < sequence_length:
            bonus[pos] = bonus_strength
            
    return bonus

def apply_spm(group_completions: list[str], group_rewards: list[float], logits, tokenizer, config) -> list[torch.Tensor]:
    """
    Ties SPM and NTHR together. To be called by grpo_train.py per training step.
    
    Args:
        group_completions: A list of G string completions for a single prompt.
        group_rewards: A list of G scalar rewards/advantages for those completions.
        logits: (Unused here, but present in signature as requested) can be used later for semantic checks.
        tokenizer: The tokenizer.
        config: A dictionary containing mask_strength, bonus_strength, etc.
        
    Returns:
        A list of G 1D tensors (token-level advantages/weights) to be multiplied 
        against the KL-penalized log probs in the GRPO loss function.
    """
    mask_strength = config.get("spm_mask_strength", 0.5)
    bonus_strength = config.get("nthr_bonus_strength", 0.1)
    
    # We define "correct" vs "incorrect" based on the reward (or advantage) being positive.
    # In GRPO, advantages are typically normalized, so > 0 is better than average.
    correct_indices = [i for i, r in enumerate(group_rewards) if r > 0]
    incorrect_indices = [i for i, r in enumerate(group_rewards) if r <= 0]
    
    correct_completions = [group_completions[i] for i in correct_indices]
    incorrect_completions = [group_completions[i] for i in incorrect_indices]
    
    # Identify overlaps
    inc_shared_positions = find_shared_prefix_tokens(correct_completions, incorrect_completions, tokenizer)
    cor_shared_positions = find_shared_correct_tokens(correct_completions, incorrect_completions, tokenizer)
    
    modified_advantages = []
    
    # To map back to the original group list
    inc_ptr = 0
    cor_ptr = 0
    
    for i, completion in enumerate(group_completions):
        # We need the token length to create the tensor.
        seq_len = len(tokenizer.encode(completion, add_special_tokens=False))
        base_advantage = group_rewards[i]
        
        # Start with a constant token-level advantage for the whole sequence
        token_advantages = torch.full((seq_len,), base_advantage, dtype=torch.float32)
        
        if i in incorrect_indices:
            # Apply SPM (Semantic Prefix Masking)
            shared_pos = inc_shared_positions[inc_ptr]
            mask = compute_soft_mask(shared_pos, seq_len, mask_strength)
            # The advantage is negative (or <= 0). We multiply it by the mask (in [0,1])
            # to dampen the negative gradient on shared prefix tokens.
            token_advantages = token_advantages * mask
            inc_ptr += 1
            
        elif i in correct_indices:
            # Apply NTHR Bonus
            shared_pos = cor_shared_positions[cor_ptr]
            bonus = apply_nthr_bonus(shared_pos, seq_len, bonus_strength)
            # Add the bonus directly to the positive advantage for shared tokens
            token_advantages = token_advantages + bonus
            cor_ptr += 1
            
        modified_advantages.append(token_advantages)
        
    return modified_advantages
