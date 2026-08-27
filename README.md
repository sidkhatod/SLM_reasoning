# prism-grpo

A project for fine-tuning small language models with a custom 3-layer extension of GRPO (Group Relative Policy Optimization), designed for a single 16GB GPU environment (Colab/Kaggle).

## Directory Structure

- `configs/`: YAML configuration files for different runs.
- `training/`: Training scripts for SFT and GRPO.
  - `layers/`: Custom layers (SPM, ICR, DSAC).
- `rewards/`: Reward functions for GRPO.
- `evaluation/`: Evaluation scripts for GSM8k, StrategyQA, MMLU.
- `inference/`: Inference scripts.
- `utils/`: Shared utilities (checkpointing, metrics, config loader, logging).

## How to Run

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```

### Training (SFT)
*Command placeholder: `python training/sft_train.py --config configs/sft.yaml`*

### Training (GRPO)
*Command placeholder: `python training/grpo_train.py --config configs/grpo_base.yaml`*

### Evaluation
*Command placeholder for evaluation scripts.*

