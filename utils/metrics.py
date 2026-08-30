"""
Telemetry for the three PRISM-GRPO diagnostics:

  * LLD Severity          - how much gradient corruption a vanilla update would cause
  * IVS                   - discriminative reasoning steps recovered per group (Layer 2)
  * Productive Zone Ratio - share of training steps drawn from the DSAC buffer (Layer 3)

plus the standard RL health signals (KL, adaptive beta, policy entropy, reward split).
"""

try:
    import wandb
except ImportError:
    wandb = None


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
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

    def log_metrics(self, step: int, lld_severity: float = None, ivs_score: float = None,
                    beta: float = None, kl: float = None, rewards: dict = None,
                    entropy: float = None, loss: float = None, extra: dict = None) -> float:
        ratio = self.get_productive_zone_ratio()

        if wandb is not None and getattr(wandb, "run", None) is not None:
            metrics = {"Productive_Zone_Ratio": ratio, "step": step}
            if lld_severity is not None:
                metrics["LLD_Severity"] = lld_severity
            if ivs_score is not None:
                metrics["Implicit_Verification_Score"] = ivs_score
            if beta is not None:
                metrics["KL_Beta"] = beta
            if kl is not None:
                metrics["KL_Divergence"] = kl
            if entropy is not None:
                metrics["Policy_Entropy"] = entropy
            if loss is not None:
                metrics["Loss"] = loss
            if rewards is not None:
                for k, v in rewards.items():
                    metrics[f"Reward_{k}"] = v
            if extra is not None:
                metrics.update(extra)

            wandb.log(metrics)
        return ratio


# Global instance for easy import across modules
_tracker = MetricTracker()


def record_curriculum_selection(from_buffer: bool):
    """Call this whenever a problem is sampled for training."""
    _tracker.record_selection(from_buffer)


def log_grpo_metrics(step: int, lld_severity: float = None, ivs_score: float = None,
                     beta: float = None, kl: float = None, rewards: dict = None,
                     entropy: float = None, loss: float = None, extra: dict = None) -> float:
    """
    Logs the running curriculum ratio alongside LLD Severity, IVS, beta, KL, entropy
    and the per-component reward split. Returns the productive zone ratio.
    """
    return _tracker.log_metrics(step, lld_severity, ivs_score, beta, kl, rewards,
                                entropy, loss, extra)


# Backwards-compatible alias: earlier code referred to this as the curriculum logger.
log_curriculum_metrics = log_grpo_metrics


def reset_metrics():
    """Clears accumulated curriculum counters (used by the tests)."""
    _tracker.reset()


# =====================================================================
# Functional metric interfaces for explicit tracking
# =====================================================================

def lld_severity_score(group_completions: list, correctness_labels: list, tokenizer) -> float:
    """
    Computes Lazy Likelihood Displacement (LLD) Severity from decoded strings:
    the fraction of tokens in incorrect completions that act as a shared prefix
    with any correct completion.

    The trainer uses `spm.compute_lld_severity` directly on the generated token ids -
    that avoids a decode/re-encode round trip which does not preserve token counts.
    This string entry point is for offline analysis.
    """
    from training.layers.spm import compute_lld_severity

    token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in group_completions]
    return compute_lld_severity(token_ids, correctness_labels)


def implicit_verification_score(step_clusters, threshold: float = 0.5) -> int:
    """
    Computes IVS (Implicit Verification Score): the number of semantic reasoning step
    clusters that are discriminative, i.e. that appear predominantly in correct
    completions rather than incorrect ones.

    Accepts either the {cluster_id: cluster_data} dict that `icr.cluster_and_align_steps`
    returns, or a plain list of cluster dicts.
    """
    if not step_clusters:
        return 0

    values = step_clusters.values() if isinstance(step_clusters, dict) else step_clusters
    return sum(1 for c in values if c.get("discriminativeness", 0.0) > threshold)


def productive_zone_ratio(curriculum_selection_log: list) -> float:
    """
    Computes Productive Zone Ratio.
    Returns the share of training steps spent on buffer-selected (True) vs random problems (False).
    """
    if not curriculum_selection_log:
        return 0.0

    buffer_count = sum(1 for x in curriculum_selection_log if x is True)
    return float(buffer_count / len(curriculum_selection_log))
