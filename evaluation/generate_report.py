"""
Aggregates every JSON in outputs/eval_results into one Markdown benchmark report.

Rows are discovered from the result files themselves rather than a hardcoded list, so
whichever of the 7 ablation conditions actually ran shows up with real numbers, and
the rest are listed explicitly as not run. Nothing here invents a number.

    python evaluation/generate_report.py
"""

import argparse
import json
import os
from collections import defaultdict

# The 7 planned ablation conditions, in the order they belong in the report.
ABLATIONS = [
    ("a0_sft_only", "A0 - SFT only", "SPM off, ICR off, DSAC off (no RL)"),
    ("a1_vanilla_grpo", "A1 - Vanilla GRPO", "SPM off, ICR off, DSAC off"),
    ("a2_grpo_spm", "A2 - GRPO + SPM", "SPM on"),
    ("a3_grpo_icr", "A3 - GRPO + ICR", "ICR on"),
    ("a4_grpo_dsac", "A4 - GRPO + DSAC", "DSAC on"),
    ("a5_grpo_spm_icr", "A5 - GRPO + SPM + ICR", "SPM on, ICR on"),
    ("a6_full_prism_grpo", "A6 - PRISM-GRPO (full)", "SPM on, ICR on, DSAC on"),
]

TASKS = [("gsm8k", "GSM8K"), ("strategyqa", "StrategyQA"), ("mmlu", "MMLU")]


def collect(results_dir):
    """Reads `{stem}__{task}.json` files into {stem: {task: metrics}}."""
    data = defaultdict(dict)
    if not os.path.isdir(results_dir):
        return data

    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".json"):
            continue
        stem_task = filename[:-len(".json")]
        if "__" not in stem_task:
            print(f"[report] Skipping {filename}: expected a '<checkpoint>__<task>.json' name.")
            continue
        stem, task = stem_task.rsplit("__", 1)
        try:
            with open(os.path.join(results_dir, filename), "r", encoding="utf-8") as f:
                data[stem][task] = json.load(f).get("metrics", {})
        except Exception as e:
            print(f"[report] Error reading {filename}: {e}")
    return data


def _match_stem(data, condition_key):
    """Finds the result stem belonging to an ablation condition, if it ran."""
    for stem in data:
        if condition_key in stem:
            return stem
    return None


def _cell(metrics, key):
    val = metrics.get(key) if metrics else None
    return f"{val * 100:.1f}%" if isinstance(val, (int, float)) else "not run"


def build_report(data, k_val):
    rows = []
    for key, label, desc in ABLATIONS:
        stem = _match_stem(data, key)
        rows.append((label, desc, stem, data.get(stem, {}) if stem else {}))

    extra_stems = [s for s in data if not any(k in s for k, _, _ in ABLATIONS)]

    md = "# PRISM-GRPO Benchmark Report\n\n"
    md += ("Generated from `outputs/eval_results/`. Cells marked **not run** have no "
           "result file - they are not estimates.\n\n")

    for title, metric_key in (("Pass@1 Accuracy (greedy decode)", "pass@1"),
                              (f"Pass@{k_val} Accuracy (greedy + {k_val} sampled)", f"pass@{k_val}")):
        md += f"### {title}\n\n"
        md += "| Condition | Layers | " + " | ".join(n for _, n in TASKS) + " | n |\n"
        md += "|---|---|" + "|".join("---" for _ in TASKS) + "|---|\n"
        for label, desc, stem, results in rows:
            cells = [_cell(results.get(t), metric_key) for t, _ in TASKS]
            n = results.get("gsm8k", {}).get("n_examples", "-") if results else "-"
            md += f"| {label} | {desc} | " + " | ".join(cells) + f" | {n} |\n"
        for stem in extra_stems:
            cells = [_cell(data[stem].get(t), metric_key) for t, _ in TASKS]
            n = data[stem].get("gsm8k", {}).get("n_examples", "-")
            md += f"| {stem} | (unregistered run) | " + " | ".join(cells) + f" | {n} |\n"
        md += "\n"

    ran = [label for label, _, stem, _ in rows if stem]
    md += "### Coverage\n\n"
    md += f"- Conditions with results: {len(ran)}/{len(ABLATIONS)}"
    md += (" - " + ", ".join(ran) if ran else " - none yet") + "\n"
    missing = [label for label, _, stem, _ in rows if not stem]
    if missing:
        md += f"- Not yet run: {', '.join(missing)}\n"
    md += ("\n**Targets:** GSM8K >= 50% and +5pts over the vanilla-GRPO baseline; "
           "StrategyQA >= 65% and +5pts over baseline. MMLU is a bonus regression "
           "guard, not a KPI.\n")
    return md


def main():
    parser = argparse.ArgumentParser(description="Build the benchmark report")
    parser.add_argument("--results_dir", type=str, default="outputs/eval_results")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data = collect(args.results_dir)
    if not data:
        print(f"No results found in {args.results_dir}. Run the eval scripts first, e.g.\n"
              "  python evaluation/eval_gsm8k.py --base_model Qwen/Qwen2.5-7B "
              "--checkpoint_path outputs/a6_full_prism_grpo/best")
        return

    k_val = 4
    for tasks in data.values():
        for metrics in tasks.values():
            for m_key in metrics:
                if m_key.startswith("pass@") and m_key != "pass@1":
                    try:
                        k_val = int(m_key.split("@")[1])
                    except ValueError:
                        pass

    md = build_report(data, k_val)
    out_path = args.out or os.path.join(args.results_dir, "benchmark_report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
