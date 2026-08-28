import torch

# Lazy load sentence transformer to avoid overhead if not used
_embedder_model = None

def get_embedder():
    global _embedder_model
    if _embedder_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using the lightweight model as requested for fast, frozen embedding extraction
            _embedder_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("Warning: sentence-transformers not installed. Using a dummy random embedder for testing.")
            class DummyEmbedder:
                def encode(self, texts, convert_to_tensor=True, show_progress_bar=False):
                    # Dummy dimensions for all-MiniLM-L6-v2
                    return torch.randn(len(texts), 384)
            _embedder_model = DummyEmbedder()
    return _embedder_model


def split_into_steps(completion_text: str) -> list[str]:
    """
    Splits a chain-of-thought completion into individual reasoning steps.
    Reuses the prompt template convention (e.g., splitting on semantic boundaries).
    """
    # We strip out standard markers if present from the SFT template
    text = completion_text.replace("Let's think step by step.", "").strip()
    
    # Split by newlines
    raw_lines = text.split('\n')
    steps = []
    
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
            
        # Split by '. ' as a simple sentence/step boundary heuristic
        sentences = line.split('. ')
        for s in sentences:
            s = s.strip()
            if s:
                # Ensure it ends with a period for consistent embeddings
                if not s.endswith('.') and not s.endswith('?') and not s.endswith('!'):
                    s += '.'
                steps.append(s)
                
    return steps


def embed_steps(steps: list[str]) -> torch.Tensor:
    """
    Embeds each step using the all-MiniLM-L6-v2 sentence encoder.
    The model is loaded once and frozen (not fine-tuned).
    """
    if not steps:
        return torch.empty((0, 384)) # Dim of all-MiniLM-L6-v2
        
    model = get_embedder()
    # Convert to tensor and move to CPU to save GPU memory 
    # since clustering happens on CPU and is tiny
    embeddings = model.encode(steps, convert_to_tensor=True, show_progress_bar=False)
    return embeddings.cpu()


def cluster_and_align_steps(group_step_embeddings: list[torch.Tensor], group_correctness_labels: list[bool], similarity_threshold: float = 0.75) -> tuple:
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
    if not any(group_correctness_labels):
        return None, None
        
    clusters = {}
    cluster_centroids = []
    completion_cluster_assignments = []
    
    next_cluster_id = 0
    
    for comp_idx, embeddings in enumerate(group_step_embeddings):
        assignments = []
        
        # embeddings is shape (num_steps, 384)
        for step_idx in range(embeddings.shape[0]):
            emb = embeddings[step_idx].unsqueeze(0) # (1, 384)
            
            best_sim = -1.0
            best_cluster_id = -1
            
            if cluster_centroids:
                centroids_tensor = torch.stack(cluster_centroids) # (num_clusters, 384)
                
                # Compute Cosine Similarity
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                centroids_norm = torch.nn.functional.normalize(centroids_tensor, p=2, dim=1)
                
                similarities = torch.mm(emb_norm, centroids_norm.transpose(0, 1)).squeeze(0)
                best_sim, best_idx = torch.max(similarities, dim=0)
                
                if best_sim.item() >= similarity_threshold:
                    best_cluster_id = best_idx.item()
                    
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
                cluster_centroids[best_cluster_id] = 0.9 * cluster_centroids[best_cluster_id] + 0.1 * emb.squeeze(0)
                
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


def compute_consensus_reward(step_clusters: dict, per_completion_step_assignments: list[list[int]]) -> dict:
    """
    Converts cluster discriminativeness scores into a per-completion reward contribution.
    
    Args:
        step_clusters: The cluster metadata containing discriminativeness scores.
        per_completion_step_assignments: List of cluster IDs assigned to each step of a completion.
        
    Returns:
        Dictionary mapping completion_id (int) to the consensus reward (float).
        If the fallback was triggered, returns a signal {"fallback": True}.
    """
    if step_clusters is None:
        return {"fallback": True}
        
    rewards = {}
    for comp_idx, cluster_ids in enumerate(per_completion_step_assignments):
        reward = 0.0
        if cluster_ids:
            # We take the average discriminativeness of the steps in this completion.
            # This rewards completions that are composed of highly discriminative, correct steps.
            sum_discrim = sum(step_clusters[cid]["discriminativeness"] for cid in cluster_ids)
            reward = sum_discrim / len(cluster_ids)
            
        rewards[comp_idx] = reward
        
    return rewards
