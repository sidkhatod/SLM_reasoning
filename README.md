# PRISM-GRPO 

**PRISM-GRPO** is a memory-conscious framework for fine-tuning small language models (like Qwen2.5-7B) using a highly customized 3-layer extension of Group Relative Policy Optimization (GRPO). It is designed from the ground up to operate within the constraints of a single 16GB GPU (e.g., Google Colab, Kaggle) utilizing 4-bit quantization, gradient checkpointing, and dynamic batching.

## Features & Custom Layers

PRISM-GRPO extends standard GRPO with three novel techniques specifically built to improve logical reasoning in small capacity models:

1. **Semantic Prefix Masking (SPM) `training/layers/spm.py`:** Prevents Lazy Likelihood Displacement by masking the negative gradients of shared valid reasoning prefixes in incorrect completions. Includes the **NTHR (Non-Trivial Hindsight Reward)** bonus.
2. **Implicit Consensus Rewards (ICR) `training/layers/icr.py`:** A step-level consensus reward that extracts discriminative reasoning clusters across a sampled group without requiring an external process-reward model, relying on a lightweight sentence encoder.
3. **Dual-Signal Adaptive Curriculum (DSAC) `training/layers/dsac.py`:** Dynamically curates the training dataset stream based on Prefix-Validity and Semantic-Uncertainty, keeping the model continuously engaged in its "productive learning zone".

## Directory Structure

```text
prism-grpo/
├── configs/              # YAML configurations for all SFT, GRPO, and Ablation runs
│   └── ablations/        # Specialized configs to independently toggle SPM, ICR, and DSAC
├── evaluation/           # Pass@1 and Pass@K evaluations (GSM8K, StrategyQA, MMLU) & Reporting
├── rewards/              # 6-Component Reward logic including Math-Shepherd PRM integration
├── training/
│   ├── layers/           # The custom SPM, ICR, and DSAC architectures
│   ├── grpo_train.py     # Main RL training loop (CustomGRPOTrainer)
│   └── sft_train.py      # Supervised Fine-Tuning warmup using multi-dataset aggregation
├── inference/            # Basic inference scripts
└── utils/                # Checkpointing (State survival) and Metrics (WandB telemetry)
```

## How to Run

### 1. Setup
Install the necessary dependencies and initialize the environment:
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. SFT Warmup
Supervised fine-tuning across GSM8K, AQuA-RAT, and StrategyQA to teach the model a consistent `<think> \n Final Answer:` pattern.
```bash
python training/sft_train.py --config configs/sft.yaml
```

### 3. PRISM-GRPO Training
Run the custom GRPO loop. The script automatically handles Adaptive KL (via PID control) and dynamic DSAC curriculum polling.
```bash
python training/grpo_train.py --config configs/grpo_base.yaml --sft_checkpoint outputs/sft_warmup
```
*(Optionally include `--resume_from outputs/my_checkpoint` to resume a pre-empted Kaggle/Colab run safely, maintaining curriculum buffer state and KL beta values.)*

### 4. Running Ablations
To isolate the contributions of specific layers, you can use the predefined config files, or toggle them via CLI flags:
```bash
# E.g., Run GRPO with ICR and DSAC disabled (SPM only)
python training/grpo_train.py --config configs/ablations/a2_grpo_spm.yaml --sft_checkpoint outputs/sft_warmup
# Alternatively, using CLI overrides:
python training/grpo_train.py --sft_checkpoint outputs/sft_warmup --no_icr --no_dsac
```

### 5. Evaluation & Benchmarking
Test your trained adapters against standard benchmarks and generate a unified Markdown report:
```bash
python evaluation/eval_gsm8k.py --base_model Qwen/Qwen2.5-7B --checkpoint_path outputs/grpo_a6_full
python evaluation/eval_strategyqa.py --base_model Qwen/Qwen2.5-7B --checkpoint_path outputs/grpo_a6_full
python evaluation/eval_mmlu.py --base_model Qwen/Qwen2.5-7B --checkpoint_path outputs/grpo_a6_full

# Auto-generate projected benchmark results across all outputs:
python evaluation/generate_report.py
```
