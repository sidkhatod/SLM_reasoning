import os
import json
import glob
from collections import defaultdict

def generate_markdown_table(data_dict, k_val=4):
    """
    data_dict: {
        "sft-only": {"gsm8k": 0.45, "strategyqa": 0.50},
        "grpo_a1": {"gsm8k": 0.55, ...},
        ...
    }
    """
    models = ["SFT-only", "GRPO A1", "GRPO A6", "GRPO A3", "Phi-3"]
    # Mapping friendly names to expected checkpoint prefixes (adjust as needed)
    model_keys = {
        "SFT-only": "sft_qwen2.5-7b",
        "GRPO A1": "grpo_a1",
        "GRPO A6": "grpo_a6",
        "GRPO A3": "grpo_a3",
        "Phi-3": "phi3"
    }
    
    datasets = ["gsm8k", "strategyqa", "mmlu"]
    dataset_names = {"gsm8k": "GSM8K", "strategyqa": "StrategyQA", "mmlu": "MMLU"}
    
    md = "# Projected Benchmark Results\n\n"
    
    # Pass@1 Table
    md += "### Pass@1 Accuracy (Greedy / First Sample)\n"
    md += "| Model | " + " | ".join([dataset_names[d] for d in datasets]) + " |\n"
    md += "|-------|" + "|".join(["---" for _ in datasets]) + "|\n"
    
    for model_name in models:
        key = model_keys[model_name]
        row = [model_name]
        for ds in datasets:
            val = data_dict.get(key, {}).get(ds, {}).get("pass@1")
            if val is not None:
                row.append(f"{val*100:.1f}%")
            else:
                row.append("n/a")
        md += "| " + " | ".join(row) + " |\n"
        
    md += "\n"
    
    # Pass@K Table
    md += f"### Pass@{k_val} Accuracy (Sampled)\n"
    md += "| Model | " + " | ".join([dataset_names[d] for d in datasets]) + " |\n"
    md += "|-------|" + "|".join(["---" for _ in datasets]) + "|\n"
    
    for model_name in models:
        key = model_keys[model_name]
        row = [model_name]
        for ds in datasets:
            val = data_dict.get(key, {}).get(ds, {}).get(f"pass@{k_val}")
            if val is not None:
                row.append(f"{val*100:.1f}%")
            else:
                row.append("n/a")
        md += "| " + " | ".join(row) + " |\n"
        
    return md

def main():
    results_dir = "outputs/eval_results"
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} does not exist. Run evaluations first.")
        return
        
    data = defaultdict(dict)
    k_val = 4 # default fallback
    
    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
            
        # expected format: {checkpoint_name}_{dataset}.json
        filepath = os.path.join(results_dir, filename)
        name_parts = filename.replace(".json", "").split("_")
        
        # Robust parsing in case checkpoint name has underscores
        dataset = name_parts[-1]
        checkpoint_name = "_".join(name_parts[:-1])
        
        with open(filepath, "r") as f:
            try:
                res = json.load(f)
                metrics = res.get("metrics", {})
                data[checkpoint_name][dataset] = metrics
                
                # Determine K dynamically from keys
                for m_key in metrics.keys():
                    if m_key.startswith("pass@") and m_key != "pass@1":
                        k_val = int(m_key.split("@")[1])
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    md_content = generate_markdown_table(data, k_val)
    
    out_path = "outputs/eval_results/benchmark_report.md"
    with open(out_path, "w") as f:
        f.write(md_content)
        
    print(md_content)
    print(f"\nReport saved to {out_path}")

if __name__ == "__main__":
    main()
