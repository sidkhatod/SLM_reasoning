import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.icr import (
    split_into_steps,
    embed_steps,
    cluster_and_align_steps,
    compute_consensus_reward
)

def test():
    # 1. Toy Completions
    # We have 3 completions: 2 correct, 1 incorrect.
    # Correct ones share a similar reasoning step: "Then we divide by 2" and "Divide both sides by two"
    # Incorrect one lacks this step.
    
    comp_correct_1 = "First we add 4 to 6. This gives 10. Then we divide by 2. Final Answer: 5."
    comp_correct_2 = "Add four to six to get 10. Divide both sides by two. Final Answer: 5."
    comp_incorrect = "Add 4 and 6 to get 10. Then we multiply by 2. Final Answer: 20."
    
    group_completions = [comp_correct_1, comp_correct_2, comp_incorrect]
    group_correctness_labels = [True, True, False]
    
    print("\n--- Test 1: split_into_steps ---")
    group_steps = []
    for i, c in enumerate(group_completions):
        steps = split_into_steps(c)
        group_steps.append(steps)
        print(f"Completion {i+1} Steps:")
        for s in steps:
            print(f"  - {s}")
            
    print("\n--- Test 2: embed_steps ---")
    print("Embedding steps using sentence-transformers (may take a moment to download on first run)...")
    group_step_embeddings = []
    for steps in group_steps:
        emb = embed_steps(steps)
        group_step_embeddings.append(emb)
    print(f"Embedded {len(group_step_embeddings)} completions.")
    print(f"Embedding shape for completion 1: {group_step_embeddings[0].shape}")
    
    print("\n--- Test 3: cluster_and_align_steps ---")
    similarity_threshold = 0.50  # Lowered for toy text to ensure matching
    clusters, assignments = cluster_and_align_steps(
        group_step_embeddings, 
        group_correctness_labels, 
        similarity_threshold=similarity_threshold
    )
    
    if clusters is None:
        print("Fallback triggered!")
    else:
        print(f"Created {len(clusters)} unique semantic clusters.")
        for cid, cdata in clusters.items():
            print(f"\nCluster {cid}:")
            print(f"  - Appears in completions: {list(cdata['completions_containing'])}")
            print(f"  - Discriminativeness Score: {cdata['discriminativeness']:.2f}")
            
            # Print the steps assigned to this cluster for debug visibility
            assigned_texts = []
            for comp_idx, comp_assignments in enumerate(assignments):
                for step_idx, assigned_cid in enumerate(comp_assignments):
                    if assigned_cid == cid:
                        assigned_texts.append(group_steps[comp_idx][step_idx])
            print(f"  - Steps: {assigned_texts}")
            
    print("\n--- Test 4: compute_consensus_reward ---")
    rewards = compute_consensus_reward(clusters, assignments)
    for comp_idx, reward in rewards.items():
        label = "Correct" if group_correctness_labels[comp_idx] else "Incorrect"
        print(f"{label} Completion {comp_idx+1} Consensus Reward: {reward:.3f}")
        
    print("\n--- Test 5: All-Negative-Group Fallback ---")
    fallback_labels = [False, False, False]
    fallback_clusters, fallback_assignments = cluster_and_align_steps(
        group_step_embeddings, 
        fallback_labels, 
        similarity_threshold=similarity_threshold
    )
    fallback_rewards = compute_consensus_reward(fallback_clusters, fallback_assignments)
    print(f"Fallback rewards output: {fallback_rewards}")

if __name__ == "__main__":
    test()
