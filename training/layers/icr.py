"""
Layer 2 - Implicit Consensus Rewards (ICR).

Extracts dense, step-level verification from the group of G completions that GRPO
already samples, at no extra generation cost. Reasoning steps are embedded with a
frozen all-MiniLM-L6-v2 encoder, clustered by cosine similarity across the group,
and scored by how *discriminative* they are: steps that show up in correct
completions but are absent from incorrect ones carry real signal.

Public entry point used by the trainer:
    compute_icr_rewards(completions, correctness_labels, config) -> (rewards, stats)
"""

import re
import torch

EMBED_DIM = 384  # all-MiniLM-L6-v2

# Lazy load sentence transformer to avoid overhead if not used
_embedder_model = None


def get_embedder():
    global _embedder_model
    if _embedder_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using the lightweight model as requested for fast, frozen embedding extraction
            _embedder_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: could not load all-MiniLM-L6-v2 ({e}). "
                  "Falling back to a deterministic hash embedder (tests only).")

            class HashEmbedder:
                """Deterministic bag-of-words hash embedder.

                Only used when sentence-transformers is unavailable. Unlike a random
                embedder it is *stable* for identical text, so clustering still
                behaves sensibly and unit tests stay reproducible.
                """

                def encode(self, texts, convert_to_tensor=True, show_progress_bar=False,
                           normalize_embeddings=False):
                    if isinstance(texts, str):
                        texts = [texts]
                    out = torch.zeros(len(texts), EMBED_DIM)
                    for i, t in enumerate(texts):
                        for tok in re.findall(r"[a-z0-9]+", t.lower()):
                            out[i, hash(tok) % EMBED_DIM] += 1.0
                    norms = out.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    return out / norms

            _embedder_model = HashEmbedder()
    return _embedder_model


# A decimal point / thousands separator must not be treated as a step boundary,
# so a '.' only splits when it is not sitting between two digits.
_SENTENCE_SPLIT = re.compile(r'(?<!\d)\.(?=\s|$)|(?<=\d)\.(?=\s|$)')


def split_into_steps(completion_text: str) -> list[str]:
    """
    Splits a chain-of-thought completion into individual reasoning steps.
    Reuses the prompt template convention (e.g., splitting on semantic boundaries).
    """
    # We strip out standard markers if present from the SFT template
    text = completion_text.replace("Let's think step by step.", "")
    # The final-answer line is an outcome, not a reasoning step - it is scored by
    # the outcome reward instead, and keeping it would give every completion a
    # trivially-common cluster that carries no discriminative information.
    text = re.split(r'Final Answer:', text, flags=re.IGNORECASE)[0].strip()

    steps = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        for s in _SENTENCE_SPLIT.split(line):
            if not s:
                continue
            s = s.strip()
            if not s:
                continue
            # Ensure it ends with a period for consistent embeddings
            if not s.endswith(('.', '?', '!')):
                s += '.'
            steps.append(s)

    return steps


def embed_steps(steps: list[str]) -> torch.Tensor:
    """
    Embeds each step using the all-MiniLM-L6-v2 sentence encoder.
    The model is loaded once and frozen (not fine-tuned).
    """
    if not steps:
        return torch.empty((0, EMBED_DIM))

    model = get_embedder()
    # Convert to tensor and move to CPU to save GPU memory
    # since clustering happens on CPU and is tiny
    embeddings = model.encode(steps, convert_to_tensor=True, show_progress_bar=False)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    return embeddings.detach().cpu().float()


def cluster_and_align_steps(group_step_embeddings: list[torch.Tensor],
                            group_correctness_labels: list[bool],
                            similarity_threshold: float = 0.75) -> tuple:
    """
    Clusters steps across the group of G completions by embedding similarity.

    Args:
        group_step_embeddings: List of (num_steps, embed_dim) tensors for each completion.
        group_correctness_labels: List of boolean labels (True if correct).
        similarity_threshold: Threshold for cosine similarity to join a cluster.

    Returns:
        tuple (step_clusters, completion_cluster_assignments) or (None, None) if fallback triggered.
    """
    # Explicit ALL-NEGATIVE-GROUP FALLBACK
    # If all G completions are incorrect, step alignment against "correct" examples is impossible.
    # The same holds for an all-positive group: nothing is discriminative when there is
    # no negative class to contrast against.
    if not any(group_correctness_labels) or all(group_correctness_labels):
        return None, None

    clusters = {}
    cluster_centroids = []
    completion_cluster_assignments = []

    next_cluster_id = 0

    for comp_idx, embeddings in enumerate(group_step_embeddings):
        assignments = []

        # embeddings is shape (num_steps, 384)
        for step_idx in range(embeddings.shape[0]):
            emb = embeddings[step_idx].unsqueeze(0)  # (1, 384)

            best_cluster_id = -1

            if cluster_centroids:
                centroids_tensor = torch.stack(cluster_centroids)  # (num_clusters, 384)

                # Compute Cosine Similarity
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                centroids_norm = torch.nn.functional.normalize(centroids_tensor, p=2, dim=1)

                similarities = torch.mm(emb_norm, centroids_norm.transpose(0, 1)).squeeze(0)
                best_sim, best_idx = torch.max(similarities, dim=0)

                if best_sim.item() >= similarity_threshold:
                    best_cluster_id = int(best_idx.item())

            if best_cluster_id == -1:
                # Create a new cluster
                best_cluster_id = next_cluster_id
                cluster_centroids.append(emb.squeeze(0))
                clusters[best_cluster_id] = {
                    "completions_containing": set(),
                }
                next_cluster_id += 1
            else:
                # Exponential moving average to slightly update centroid
                cluster_centroids[best_cluster_id] = (
                    0.9 * cluster_centroids[best_cluster_id] + 0.1 * emb.squeeze(0)
                )

            # Record that this completion contains a step from this cluster
            clusters[best_cluster_id]["completions_containing"].add(comp_idx)
            assignments.append(best_cluster_id)

        completion_cluster_assignments.append(assignments)

    # Calculate Discriminativeness Score for each cluster
    total_correct = max(1, sum(1 for l in group_correctness_labels if l))
    total_incorrect = max(1, sum(1 for l in group_correctness_labels if not l))

    for cid, cdata in clusters.items():
        correct_count = sum(1 for idx in cdata["completions_containing"] if group_correctness_labels[idx])
        incorrect_count = sum(1 for idx in cdata["completions_containing"] if not group_correctness_labels[idx])

        freq_correct = correct_count / total_correct
        freq_incorrect = incorrect_count / total_incorrect

        # Highly discriminative steps appear in correct completions but NOT in incorrect ones
        cdata["correct_count"] = correct_count
        cdata["incorrect_count"] = incorrect_count
        cdata["discriminativeness"] = freq_correct - freq_incorrect

    return clusters, completion_cluster_assignments


def compute_consensus_reward(step_clusters, per_completion_step_assignments,
                             num_completions: int = None) -> list[float]:
    """
    Converts cluster discriminativeness scores into a per-completion reward contribution.

    Discriminativeness lives in [-1, 1]; the returned reward is rescaled to [0, 1] so
    that it composes linearly with the other five reward components, all of which are
    also [0, 1]. A completion with no recognised steps, or a group that hit the
    all-negative fallback, scores the neutral 0.5.

    Returns:
        List of G floats (one per completion).
    """
    if num_completions is None:
        num_completions = len(per_completion_step_assignments or [])

    # Fallback: no discriminative structure available -> neutral, so the consensus
    # term contributes a constant and the outcome term drives the advantage.
    if step_clusters is None or per_completion_step_assignments is None:
        return [0.5] * num_completions

    rewards = []
    for comp_idx in range(num_completions):
        cluster_ids = (per_completion_step_assignments[comp_idx]
                       if comp_idx < len(per_completion_step_assignments) else [])
        if cluster_ids:
            # We take the average discriminativeness of the steps in this completion.
            # This rewards completions that are composed of highly discriminative, correct steps.
            sum_discrim = sum(step_clusters[cid]["discriminativeness"] for cid in cluster_ids)
            raw = sum_discrim / len(cluster_ids)
        else:
            raw = 0.0
        rewards.append(float(min(1.0, max(0.0, (raw + 1.0) / 2.0))))

    return rewards


def implicit_verification_score(step_clusters, threshold: float = 0.5) -> int:
    """
    IVS: number of clusters that are strongly discriminative for this group.
    Counts clusters whose discriminativeness exceeds `threshold` (default 0.5),
    i.e. steps that appear in most correct completions and few incorrect ones.
    """
    if not step_clusters:
        return 0
    return sum(1 for c in step_clusters.values() if c.get("discriminativeness", 0.0) > threshold)


def compute_icr_rewards(completions: list[str],
                        correctness_labels: list[bool],
                        config: dict = None) -> tuple[list[float], dict]:
    """
    End-to-end ICR entry point called once per GRPO group.

    Args:
        completions: The G decoded completion strings.
        correctness_labels: G booleans - True where the completion's final answer is correct.
        config: optional dict; reads `icr_similarity_threshold` (default 0.75) and
                `icr_discriminative_threshold` (default 0.5).

    Returns:
        (rewards, stats) where rewards is a list of G floats in [0, 1] and stats
        carries the telemetry the trainer logs (IVS, cluster count, fallback flag).
    """
    config = config or {}
    sim_thresh = float(config.get("icr_similarity_threshold", 0.75))
    disc_thresh = float(config.get("icr_discriminative_threshold", 0.5))

    group_steps = [split_into_steps(c) for c in completions]
    group_embeddings = [embed_steps(s) for s in group_steps]

    clusters, assignments = cluster_and_align_steps(
        group_embeddings, correctness_labels, similarity_threshold=sim_thresh
    )

    rewards = compute_consensus_reward(clusters, assignments, num_completions=len(completions))

    stats = {
        "fallback": clusters is None,
        "num_clusters": 0 if clusters is None else len(clusters),
        "discriminative_clusters": implicit_verification_score(clusters, disc_thresh),
        "mean_steps_per_completion": (
            sum(len(s) for s in group_steps) / max(1, len(group_steps))
        ),
    }
    return rewards, stats
