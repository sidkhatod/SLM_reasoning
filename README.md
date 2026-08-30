# PRISM-GRPO

**Progressive Reasoning through Implicit Semantic Modeling** — a three-layer extension of Group
Relative Policy Optimization for small language models. Built for QLoRA fine-tuning of Qwen 2.5 7B
under constrained VRAM, with **zero inference-time overhead**: all three layers act during training
only, so serving is a plain base model plus one LoRA adapter.

See [`writeup.md`](writeup.md) for the full motivation, method and projected results.

> **Status:** the pipeline is implemented and runs end-to-end, but no full-scale training run has
> been executed. Every benchmark number in `writeup.md` is a **projection**. The report generator
> prints "not run" for conditions with no result file and never fills a cell with an estimate.

## The three layers

| Layer | Problem it solves | Where it acts |
|---|---|---|
| **SPM** — Semantic Prefix Masking (`training/layers/spm.py`) | Gradient corruption (Lazy Likelihood Displacement): negative updates on wrong completions suppress the *correct* reasoning prefix they share with right ones | Per-token advantages, before the policy loss |
| **ICR** — Implicit Consensus Rewards (`training/layers/icr.py`) | Sparse reward: outcome-only rewards give zero signal on partially-correct completions | Clusters reasoning steps across the existing `G=8` group — no extra generation, no verifier model |
| **DSAC** — Dual-Signal Adaptive Curriculum (`training/layers/dsac.py`) | Curriculum blindness: problems that are too easy collapse entropy, too hard give no signal | Curates the training stream, refreshed every 50 steps |

SPM additionally applies an **NTHR** bonus: a small positive signal on the shared prefix of the
*correct* trajectory, countering residual suppression from the failing ones.

### The 6-component reward

`0.40` outcome · `0.20` PRM (Math-Shepherd) · `0.20` consensus (ICR) · `0.10` consistency ·
`0.05` self-correction · `0.05` format.

Every component is scored in `[0, 1]`, so the weighted total is too. Format is pinned at `0.05`
deliberately — weighting it higher produces reward hacking, where the model perfects the CoT
scaffolding while reasoning quality degrades. Disabling a component (`use_prm: false`,
`use_icr: false`) redistributes its weight across the rest, keeping ablations comparable.

## Layout

```text
configs/
  grpo_base.yaml        # every GRPO hyperparameter, fully specified
  sft.yaml              # Phase 1 warmup
  ablations/            # 7 conditions + a Phi-3 cross-model check; each `extends` the base
evaluation/
  harness.py            # shared: greedy pass@1 + sampled pass@k, one answer matcher
  eval_{gsm8k,strategyqa,mmlu}.py
  generate_report.py    # aggregates results into one Markdown table
rewards/reward_functions.py   # the 6 components + answer normalisation
training/
  layers/{spm,icr,dsac}.py
  sft_train.py          # Phase 1
  grpo_train.py         # Phase 2 — PrismGRPOTrainer
inference/inference.py  # single-shot, interactive, or adapter merge
utils/
  data.py               # shared dataset pipeline + prompt template
  metrics.py            # LLD Severity, IVS, Productive Zone Ratio, KL, entropy
  checkpointing.py      # adapter + optimizer + curriculum-buffer state
tests/                  # pytest — layers, rewards, configs, end-to-end loop
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional: WANDB_API_KEY, HF_TOKEN
pytest                             # 84 tests, no GPU or model download needed
```

## 1. SFT warmup

Teaches the `Question: ... / Answer: Let's think step by step. ... / Final Answer: <value>` template
that the format reward, the ICR step splitter and the evaluation answer extractor all parse. Loss is
on the completion tokens only.

```bash
python training/sft_train.py --config configs/sft.yaml
```

## 2. PRISM-GRPO

```bash
python training/grpo_train.py --config configs/grpo_base.yaml \
    --sft_checkpoint outputs/sft_qwen2.5-7b
```

Handles adaptive KL (PID-style, `beta` initialised at 0.04 and clamped to `[0.01, 0.3]`), DSAC buffer
refreshes, and **validation-based checkpoint selection** — the best checkpoint by held-out accuracy is
written to `outputs/<run>/best`, which is the one to evaluate. Selecting the final training step
instead is how a reward-hacked policy gets shipped.

Resume a pre-empted Colab/Kaggle run with `--resume_from outputs/<run>/checkpoint-N`; the LoRA
weights, optimizer state, current `beta` and the curriculum buffer all survive.

### Ablations

Seven conditions isolate each layer and each pairing:

```bash
python training/grpo_train.py --config configs/ablations/a2_grpo_spm.yaml --sft_checkpoint outputs/sft_qwen2.5-7b
# or toggle from the CLI
python training/grpo_train.py --sft_checkpoint outputs/sft_qwen2.5-7b --no_icr --no_dsac
```

`--no_spm`, `--no_icr`, `--no_dsac`, `--no_prm` all work. `a0_sft_only.yaml` sets `skip_grpo: true`
and exits without training — evaluate the SFT checkpoint directly.

`a7_phi3_cross_model.yaml` runs the full stack on Phi-3-Mini 3.8B to check the architecture is not
Qwen-specific (Phi-3 fuses its qkv/gate-up projections, so it needs different LoRA targets).

## 3. Evaluation

```bash
CKPT=outputs/a6_full_prism_grpo/best
python evaluation/eval_gsm8k.py      --base_model Qwen/Qwen2.5-7B --checkpoint_path $CKPT
python evaluation/eval_strategyqa.py --base_model Qwen/Qwen2.5-7B --checkpoint_path $CKPT
python evaluation/eval_mmlu.py       --base_model Qwen/Qwen2.5-7B --checkpoint_path $CKPT
python evaluation/generate_report.py
```

`--limit N` caps the example count for a quick check; omitting `--checkpoint_path` evaluates the raw
base model, which is how you get the baseline row.

## 4. Inference

```bash
python inference/inference.py --checkpoint_path outputs/a6_full_prism_grpo/best \
    --question "Natalia sold clips to 48 friends in April, and half as many in May. How many altogether?"

python inference/inference.py --checkpoint_path outputs/a6_full_prism_grpo/best --interactive

# merge the adapter into the base weights for standalone deployment
python inference/inference.py --checkpoint_path outputs/a6_full_prism_grpo/best \
    --merge_and_save outputs/prism_merged
```

## Fitting the hardware

`configs/grpo_base.yaml` targets 2× 24GB. The knobs, in the order worth turning:

| Constraint | Change |
|---|---|
| PRM will not co-reside with the policy | `use_prm: false` (its 0.20 weight is redistributed), or `prm_unload_after_scoring: true` to trade a model reload per step for the VRAM |
| Single 16GB card | `group_size: 4`, `max_new_tokens: 256`, `groups_per_step: 1` |
| Still tight | `lora_r: 32`, `max_prompt_length: 256` |
| No CUDA at all | The trainers fall back to fp32 without bitsandbytes — usable only for smoke tests |

DSAC's `dsac_candidate_pool` must stay above `dsac_buffer_size`, or the curriculum keeps every
candidate it scores and silently degenerates into uniform sampling. The layer widens the pool and
warns if a config gets this wrong, and a test enforces it.

## Metrics

Logged every step to W&B (and stdout without it):

- **LLD Severity** — fraction of tokens in wrong completions sitting inside a prefix shared with a
  correct one, i.e. exactly what a vanilla negative gradient would corrupt.
- **IVS** — discriminative step clusters recovered per group. `ICR_Fallback_Rate` tracks all-negative
  groups, where step alignment is impossible and the reward falls back to outcome-driven.
- **Productive Zone Ratio** — share of training problems drawn from the DSAC buffer.
- Plus KL against the reference policy, adaptive `beta`, policy entropy (the entropy-collapse guard),
  group accuracy, and the per-component reward split.
