try:
    import wandb
except ImportError:
    wandb = None

class MetricTracker:
    def __init__(self):
        self.buffer_selected_count = 0
        self.random_selected_count = 0
        self.curriculum_selection_log = []

    def record_selection(self, from_buffer: bool):
        self.curriculum_selection_log.append(from_buffer)
        if from_buffer:
            self.buffer_selected_count += 1
        else:
            self.random_selected_count += 1
            
    def get_productive_zone_ratio(self) -> float:
        total = self.buffer_selected_count + self.random_selected_count
        if total == 0:
            return 0.0
        return self.buffer_selected_count / total
        
    def log_metrics(self, step: int, lld_severity: float = None, ivs_score: float = None, beta: float = None, kl: float = None, rewards: dict = None):
        ratio = self.get_productive_zone_ratio()
        
        if wandb is not None and wandb.run is not None:
            metrics = {"Productive_Zone_Ratio": ratio, "step": step}
            if lld_severity is not None:
                metrics["LLD_Severity"] = lld_severity
            if ivs_score is not None:
                metrics["Implicit_Verification_Score"] = ivs_score
            if beta is not None:
                metrics["KL_Beta"] = beta
            if kl is not None:
                metrics["KL_Divergence"] = kl
            if rewards is not None:
                for k, v in rewards.items():
                    metrics[f"Reward_{k}"] = v
                    
            wandb.log(metrics)
        return ratio

# Global instance for easy import across modules
_tracker = MetricTracker()

def record_curriculum_selection(from_buffer: bool):
    """
    Call this whenever a problem is sampled for training.
    """
    _tracker.record_selection(from_buffer)

def log_grpo_metrics(step: int, lld_severity: float = None, ivs_score: float = None, beta: float = None, kl: float = None, rewards: dict = None):
    """
    Call this to log the running curriculum ratio, along with LLD Severity, IVS, Beta, and rewards.
    """
    return _tracker.log_metrics(step, lld_severity, ivs_score, beta, kl, rewards)

# =====================================================================
# Finalized Functional Metric Interfaces requested for explicit tracking
# =====================================================================

def lld_severity_score(group_completions: list, correctness_labels: list, tokenizer) -> float:
    """
    Computes Lazy Likelihood Displacement (LLD) Severity.
    LLD Severity is the fraction of tokens in incorrect completions that act as a shared prefix 
    with any correct completion.
    """
    correct_completions = [c for c, label in zip(group_completions, correctness_labels) if label]
    incorrect_completions = [c for c, label in zip(group_completions, correctness_labels) if not label]
    
    if not correct_completions or not incorrect_completions:
        return 0.0
        
    correct_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in correct_completions]
    incorrect_tokens = [tokenizer.encode(c, add_special_tokens=False) for c in incorrect_completions]
    
    total_incorrect_tokens = 0
    total_shared_prefix_tokens = 0
    
    for inc_toks in incorrect_tokens:
        shared_pos = set()
        for cor_toks in correct_tokens:
            max_len = min(len(inc_toks), len(cor_toks))
            for i in range(max_len):
                if inc_toks[i] == cor_toks[i]:
                    shared_pos.add(i)
                else:
                    break
        total_incorrect_tokens += len(inc_toks)
        total_shared_prefix_tokens += len(shared_pos)
        
    if total_incorrect_tokens == 0:
        return 0.0
        
    return float(total_shared_prefix_tokens / total_incorrect_tokens)

def implicit_verification_score(step_clusters: list) -> int:
    """
    Computes IVS (Implicit Verification Score) by counting the number of semantic reasoning 
    step clusters that are discriminative (i.e. predominantly appear in correct completions 
    vs incorrect completions).
    """
    discriminative_count = 0
    # Expected step_clusters format from ICR: 
    # [{"steps": [...], "discriminativeness": 0.85, ...}, ...]
    for cluster in step_clusters:
        # A cluster is considered discriminative if its score > threshold (e.g. 0.5)
        # using the ICR heuristic.
        score = cluster.get("discriminativeness", 0.0)
        if score > 0.0:
            discriminative_count += 1
            
    return discriminative_count

def productive_zone_ratio(curriculum_selection_log: list) -> float:
    """
    Computes Productive Zone Ratio.
    Returns the share of training steps spent on buffer-selected (True) vs random problems (False).
    """
    if not curriculum_selection_log:
        return 0.0
        
    buffer_count = sum(1 for x in curriculum_selection_log if x is True)
    return float(buffer_count / len(curriculum_selection_log))
