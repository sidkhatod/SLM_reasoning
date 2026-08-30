"""
Phase 2: PRISM-GRPO training.

Implements the GRPO objective directly (PEFT + plain PyTorch) rather than subclassing
`trl.GRPOTrainer`. That is deliberate: TRL's GRPO implementation broadcasts a single
scalar advantage across every token of a completion, and Layer 1 (SPM) needs
*per-token* advantages to dampen the negative gradient on shared valid prefixes. The
loss below is the standard DeepSeek GRPO objective - group-normalised advantages, a
PPO clip, and a k3 KL estimate against a reference policy - with the three PRISM
layers injected at the points where they belong.

The reference policy is the frozen base model, obtained by disabling the LoRA adapter
(`model.disable_adapter()`). No second copy of the 7B weights is held in VRAM.

Usage:
    python training/grpo_train.py --config configs/grpo_base.yaml \
        --sft_checkpoint outputs/sft_qwen2.5-7b
"""

import argparse
import json
import math
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rewards.reward_functions import (
    answers_match,
    compute_total_reward,
    extract_final_answer,
    resolve_weights,
)
from training.layers.dsac import CurriculumBuffer, RandomBuffer
from training.layers.icr import compute_icr_rewards
from training.layers.spm import apply_spm, compute_lld_severity
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.logger import configure_console
from utils.data import load_reasoning_dataset, train_val_split
from utils.metrics import log_grpo_metrics

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_completion(token_ids: list[int], eos_id: int, pad_id: int) -> list[int]:
    """Cuts a generated row at its first EOS (kept) and strips trailing padding."""
    out = []
    for t in token_ids:
        out.append(t)
        if t == eos_id:
            return out
    while out and out[-1] == pad_id:
        out.pop()
    return out


def _token_logprobs_and_entropy(logits: torch.Tensor, target_ids: torch.Tensor):
    """
    Log-probability of each target token plus the per-position policy entropy.

    `logits` are already shifted so position i predicts target_ids[i].
    """
    logits = logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
    return token_logprobs, entropy


class PrismGRPOTrainer:
    """The PRISM-GRPO training loop: SPM + ICR + DSAC on top of GRPO."""

    def __init__(self, model, tokenizer, config, train_dataset, val_dataset,
                 dsac_buffer, output_dir, run_name):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dsac_buffer = dsac_buffer
        self.output_dir = output_dir
        self.run_name = run_name

        self.group_size = int(config.get("group_size", 8))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.groups_per_step = int(config.get("groups_per_step", 1))
        self.max_steps = int(config.get("max_steps", 1000))
        self.temperature = float(config.get("temperature", 0.9))
        self.clip_eps = float(config.get("clip_epsilon", 0.2))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))

        self.use_spm = bool(config.get("use_spm", True))
        self.use_icr = bool(config.get("use_icr", True))

        kl_cfg = config.get("adaptive_kl", {}) or {}
        self.target_kl = float(kl_cfg.get("target_kl", config.get("target_kl", 0.1)))
        self.current_beta = float(kl_cfg.get("beta_init", 0.04))
        self.beta_min = float(kl_cfg.get("beta_min", 0.01))
        self.beta_max = float(kl_cfg.get("beta_max", 0.3))
        self.kl_kp = float(kl_cfg.get("kp", 0.1))

        self.eval_steps = int(config.get("eval_steps", 100))
        self.eval_samples = int(config.get("eval_samples", 100))
        self.save_steps = int(config.get("save_steps", 100))

        self.step_counter = 0
        self.best_val_accuracy = -1.0
        self.best_step = -1

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=float(config.get("learning_rate", 1e-5)))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, int(config.get("warmup_steps", 20))))
        )

        self._warned = set()

    # -- generation --------------------------------------------------------

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _generate_group(self, prompt: str):
        """Samples G completions and returns (prompt_ids, list of completion id lists, texts)."""
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.config.get("max_prompt_length", 512))
        enc = {k: v.to(self.device) for k, v in enc.items()}

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.group_size,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=float(self.config.get("top_p", 0.95)),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if was_training:
                self.model.train()

        prompt_ids = enc["input_ids"][0]
        prompt_len = prompt_ids.shape[0]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        completion_ids = []
        for row in out[:, prompt_len:].tolist():
            trimmed = _trim_completion(row, eos_id, pad_id)
            completion_ids.append(trimmed if trimmed else [eos_id])

        texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return prompt_ids, completion_ids, texts

    # -- log-probs ---------------------------------------------------------

    def _sequence_logprobs(self, prompt_ids: torch.Tensor, completion_ids: list[int],
                           with_grad: bool, reference: bool = False):
        """
        Log-probs of the completion tokens under the current policy, or under the
        reference policy (the base model with the LoRA adapter disabled).
        """
        comp = torch.tensor(completion_ids, device=self.device, dtype=torch.long)
        full = torch.cat([prompt_ids, comp]).unsqueeze(0)
        prompt_len = prompt_ids.shape[0]

        def _forward():
            logits = self.model(full, use_cache=False).logits
            shifted = logits[:, prompt_len - 1:-1, :].squeeze(0)
            return _token_logprobs_and_entropy(shifted, comp)

        if reference:
            with torch.no_grad():
                disable = getattr(self.model, "disable_adapter", None)
                if disable is None:
                    if "ref" not in self._warned:
                        print("[GRPO] Model exposes no adapter toggle; using the current "
                              "policy as its own reference (KL will read ~0).")
                        self._warned.add("ref")
                    return _forward()
                with disable():
                    return _forward()

        if with_grad:
            return _forward()
        with torch.no_grad():
            return _forward()

    # -- one optimisation step --------------------------------------------

    def _process_group(self, example: dict):
        """Scores one problem's generation group and returns its loss plus telemetry."""
        prompt = example.get("prompt") or example.get("question", "")
        ground_truth = example.get("answer", "")

        prompt_ids, completion_ids, texts = self._generate_group(prompt)

        # Correctness labels come from the *verifiable* outcome, not from relative
        # reward - the ICR/SPM contrast is only meaningful against real correctness.
        correctness = [answers_match(extract_final_answer(t), ground_truth) for t in texts]

        # Layer 2: ICR
        if self.use_icr:
            icr_rewards, icr_stats = compute_icr_rewards(texts, correctness, self.config)
        else:
            icr_rewards, icr_stats = None, {"fallback": True, "num_clusters": 0,
                                            "discriminative_clusters": 0}
            if "icr" not in self._warned:
                print("[Ablation] ICR disabled; its consensus weight is redistributed "
                      "across the remaining reward components.")
                self._warned.add("icr")

        total_rewards, reward_components = compute_total_reward(
            prompt, texts, ground_truth, {"icr_rewards": icr_rewards}, self.config
        )

        rewards = torch.tensor(total_rewards, device=self.device, dtype=torch.float32)
        # Group-normalised advantages - the "group relative" in GRPO.
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

        lld_severity = compute_lld_severity(completion_ids, correctness)

        # Layer 1: SPM (per-token advantages)
        if self.use_spm:
            token_advantages = apply_spm(completion_ids, advantages, self.config,
                                         correctness_labels=correctness)
        else:
            token_advantages = [torch.full((len(ids),), float(advantages[i]),
                                           dtype=torch.float32)
                                for i, ids in enumerate(completion_ids)]
            if "spm" not in self._warned:
                print("[Ablation] SPM disabled; token advantages are uniform per sequence.")
                self._warned.add("spm")

        # Policy update over the group, microbatched one sequence at a time so a
        # G=8 x 512-token group fits alongside the 4-bit base weights.
        group_loss = torch.zeros((), device=self.device)
        kl_sum, entropy_sum, n_tokens = 0.0, 0.0, 0

        for i, ids in enumerate(completion_ids):
            adv = token_advantages[i].to(self.device)

            ref_logprobs, _ = self._sequence_logprobs(prompt_ids, ids, with_grad=False,
                                                      reference=True)
            new_logprobs, entropy = self._sequence_logprobs(prompt_ids, ids, with_grad=True)

            # One inner epoch per generation batch, so the behaviour policy *is* the
            # current policy: old_logprobs = new_logprobs.detach() and the ratio starts
            # at exactly 1. The clip still bounds the update, and taking the detached
            # copy here rather than a second forward pass saves a third of the compute.
            old_logprobs = new_logprobs.detach()

            ratio = torch.exp(new_logprobs - old_logprobs)
            pg = torch.max(
                -adv * ratio,
                -adv * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps),
            )

            # k3 KL estimator against the reference policy: unbiased and non-negative.
            log_diff = ref_logprobs.detach() - new_logprobs
            kl = torch.exp(log_diff) - log_diff - 1.0

            seq_loss = (pg + self.current_beta * kl).mean()
            group_loss = group_loss + seq_loss

            kl_sum += float(kl.mean().item())
            entropy_sum += float(entropy.mean().item())
            n_tokens += len(ids)

        group_loss = group_loss / max(1, len(completion_ids))

        stats = {
            "kl": kl_sum / max(1, len(completion_ids)),
            "entropy": entropy_sum / max(1, len(completion_ids)),
            "lld_severity": lld_severity,
            "ivs": icr_stats.get("discriminative_clusters", 0),
            "icr_fallback": bool(icr_stats.get("fallback", False)),
            "rewards": reward_components,
            "accuracy": sum(correctness) / max(1, len(correctness)),
            "mean_completion_tokens": n_tokens / max(1, len(completion_ids)),
        }
        return group_loss, stats

    def _update_beta(self, observed_kl: float):
        """PID-style (proportional) adaptive KL control, bounded to [beta_min, beta_max]."""
        error = observed_kl - self.target_kl
        self.current_beta = self.current_beta * math.exp(self.kl_kp * error)
        self.current_beta = max(self.beta_min, min(self.beta_max, self.current_beta))

    # -- validation --------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, limit: int = None) -> float:
        """Greedy pass@1 on the held-out validation slice."""
        if not self.val_dataset:
            return 0.0

        limit = limit or self.eval_samples
        subset = self.val_dataset[:limit]

        was_training = self.model.training
        self.model.eval()
        correct = 0
        try:
            for ex in subset:
                enc = self.tokenizer(ex["prompt"], return_tensors="pt", truncation=True,
                                     max_length=self.config.get("max_prompt_length", 512))
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                text = self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                if answers_match(extract_final_answer(text), ex["answer"]):
                    correct += 1
        finally:
            if was_training:
                self.model.train()

        return correct / max(1, len(subset))

    # -- main loop ---------------------------------------------------------

    def train(self):
        self.model.train()
        os.makedirs(self.output_dir, exist_ok=True)
        start = self.step_counter

        for step in range(start + 1, self.max_steps + 1):
            self.step_counter = step
            t0 = time.time()

            self.optimizer.zero_grad(set_to_none=True)
            agg = {"kl": 0.0, "entropy": 0.0, "lld_severity": 0.0, "ivs": 0.0,
                   "accuracy": 0.0, "loss": 0.0}
            reward_agg = {}
            fallbacks = 0

            for _ in range(self.groups_per_step):
                example = self.dsac_buffer.sample_problem(self.train_dataset)
                loss, stats = self._process_group(example)
                (loss / self.groups_per_step).backward()

                agg["loss"] += float(loss.item()) / self.groups_per_step
                for k in ("kl", "entropy", "lld_severity", "ivs", "accuracy"):
                    agg[k] += stats[k] / self.groups_per_step
                fallbacks += int(stats["icr_fallback"])
                for k, v in stats["rewards"].items():
                    reward_agg[k] = reward_agg.get(k, 0.0) + v / self.groups_per_step

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Adaptive KL: raise beta when the policy drifts past target, lower it when
            # it is over-constrained. Static beta reliably collapsed or over-constrained.
            self._update_beta(agg["kl"])

            log_grpo_metrics(
                step=step,
                lld_severity=agg["lld_severity"],
                ivs_score=agg["ivs"],
                beta=self.current_beta,
                kl=agg["kl"],
                rewards=reward_agg,
                entropy=agg["entropy"],
                loss=agg["loss"],
                extra={"Group_Accuracy": agg["accuracy"],
                       "ICR_Fallback_Rate": fallbacks / max(1, self.groups_per_step),
                       "Learning_Rate": self.lr_scheduler.get_last_lr()[0]},
            )

            if step % int(self.config.get("logging_steps", 10)) == 0:
                mem = (torch.cuda.memory_allocated() / (1024 ** 3)
                       if torch.cuda.is_available() else 0.0)
                print(f"Step {step}/{self.max_steps} | loss {agg['loss']:.4f} | "
                      f"KL {agg['kl']:.4f} | beta {self.current_beta:.4f} | "
                      f"acc {agg['accuracy']:.2f} | LLD {agg['lld_severity']:.3f} | "
                      f"IVS {agg['ivs']:.1f} | H {agg['entropy']:.3f} | "
                      f"{time.time() - t0:.1f}s | VRAM {mem:.2f}GB")

            # Layer 3: refresh the curriculum so selection tracks the live policy.
            self.dsac_buffer.maybe_refresh(step, self.model, self.tokenizer,
                                           self.train_dataset)

            # Checkpoint on validation accuracy, never on the final training step -
            # the guard against shipping a reward-hacked policy.
            if self.eval_steps > 0 and step % self.eval_steps == 0:
                acc = self.evaluate()
                print(f"[eval] step {step}: validation pass@1 = {acc * 100:.2f}% "
                      f"(best {max(acc, self.best_val_accuracy) * 100:.2f}%)")
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log({"Validation_Accuracy": acc, "step": step})
                if acc > self.best_val_accuracy:
                    self.best_val_accuracy = acc
                    self.best_step = step
                    self.save(os.path.join(self.output_dir, "best"),
                              note={"validation_accuracy": acc})
                    print(f"[eval] New best checkpoint saved at step {step}.")

            if self.save_steps > 0 and step % self.save_steps == 0:
                self.save(os.path.join(self.output_dir, f"checkpoint-{step}"))

        return self.best_val_accuracy

    def save(self, path: str, note: dict = None):
        extra = {
            "beta": self.current_beta,
            "dsac_buffer": self.dsac_buffer.buffer,
            "best_val_accuracy": self.best_val_accuracy,
            "best_step": self.best_step,
        }
        if note:
            extra.update(note)
        save_checkpoint(self.model, self.optimizer, self.lr_scheduler,
                        self.step_counter, extra, path)
        with open(os.path.join(path, "prism_state.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in extra.items() if k != "dsac_buffer"}, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_model(config, sft_checkpoint, device_map="auto"):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["base_model"]
    tokenizer_src = sft_checkpoint if (sft_checkpoint and os.path.isdir(sft_checkpoint)) else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = {"trust_remote_code": True, "device_map": device_map}
    if config.get("quantization", "4bit") == "4bit" and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32 if not torch.cuda.is_available() else torch.bfloat16

    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if "quantization_config" in load_kwargs:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.get("gradient_checkpointing", True)
        )

    adapter_present = sft_checkpoint and any(
        os.path.exists(os.path.join(sft_checkpoint, f))
        for f in ("adapter_model.bin", "adapter_model.safetensors")
    )
    if adapter_present:
        # Continue training the SFT adapter rather than stacking a fresh one on top.
        print(f"Loading SFT adapter from {sft_checkpoint}...")
        model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)
    else:
        if sft_checkpoint:
            print(f"No adapter found in {sft_checkpoint}; initialising a fresh LoRA adapter.")
        lora_config = LoraConfig(
            r=config.get("lora_r", 64),
            lora_alpha=config.get("lora_alpha", 128),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules",
                                      ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="PRISM-GRPO (Phase 2) training")
    parser.add_argument("--config", type=str, default="configs/grpo_base.yaml")
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--dataset_limit", type=int, default=None,
                        help="Cap the training pool (useful for smoke tests)")
    parser.add_argument("--no_wandb", action="store_true")

    for layer in ("spm", "icr", "dsac", "prm"):
        parser.add_argument(f"--use_{layer}", action="store_true", default=None,
                            dest=f"use_{layer}")
        parser.add_argument(f"--no_{layer}", action="store_false", dest=f"use_{layer}")

    args = parser.parse_args()

    config = load_config(args.config)

    if config.get("skip_grpo"):
        print(f"Config {args.config} sets `skip_grpo: true` (this is the SFT-only "
              "ablation). Nothing to train - evaluate the SFT checkpoint directly.")
        return

    if args.group_size is not None:
        config["group_size"] = args.group_size
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    for layer in ("spm", "icr", "dsac", "prm"):
        val = getattr(args, f"use_{layer}")
        if val is not None:
            config[f"use_{layer}"] = val

    run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
    output_dir = args.output_dir or f"outputs/{run_name}"

    print(f"Initializing PRISM-GRPO: {run_name}")
    print(f"  layers -> SPM={config.get('use_spm', True)} "
          f"ICR={config.get('use_icr', True)} DSAC={config.get('use_dsac', True)} "
          f"PRM={config.get('use_prm', True)}")
    print(f"  effective reward weights -> "
          f"{ {k: round(v, 3) for k, v in resolve_weights(config).items()} }")

    if wandb is not None and not args.no_wandb and os.getenv("WANDB_API_KEY"):
        wandb.init(project=config.get("wandb_project", "prism-grpo"),
                   name=run_name, config=config)
    elif not args.no_wandb:
        print("WANDB_API_KEY not set - metrics will print to stdout only.")

    model, tokenizer = build_model(config, args.sft_checkpoint)

    dataset = load_reasoning_dataset(
        names=config.get("datasets", ["gsm8k", "aqua_rat", "strategyqa"]),
        split="train",
    )
    if args.dataset_limit:
        dataset = dataset[:args.dataset_limit]

    train_dataset, val_dataset = train_val_split(
        dataset, val_size=int(config.get("val_size", 200))
    )
    print(f"Train pool: {len(train_dataset)} | Validation slice: {len(val_dataset)}")

    if config.get("use_dsac", True):
        dsac_buffer = CurriculumBuffer(config)
        dsac_buffer.refresh(model, tokenizer, train_dataset)
    else:
        print("[Ablation] DSAC disabled. Using uniform random sampling.")
        dsac_buffer = RandomBuffer(config)

    trainer = PrismGRPOTrainer(
        model=model, tokenizer=tokenizer, config=config,
        train_dataset=train_dataset, val_dataset=val_dataset,
        dsac_buffer=dsac_buffer, output_dir=output_dir, run_name=run_name,
    )

    if args.resume_from:
        step, extra = load_checkpoint(model, trainer.optimizer, trainer.lr_scheduler,
                                      args.resume_from)
        if step > 0:
            trainer.step_counter = step
            trainer.current_beta = extra.get("beta", trainer.current_beta)
            trainer.best_val_accuracy = extra.get("best_val_accuracy", -1.0)
            trainer.best_step = extra.get("best_step", -1)
            if extra.get("dsac_buffer"):
                dsac_buffer.buffer = extra["dsac_buffer"]
            print(f"Resumed at step {step} with beta {trainer.current_beta:.4f} "
                  f"and a {len(dsac_buffer.buffer)}-problem curriculum buffer.")

    print("Starting GRPO training...")
    best = trainer.train()

    trainer.save(os.path.join(output_dir, "final"))
    print(f"Training complete. Best validation pass@1: {best * 100:.2f}% "
          f"at step {trainer.best_step}.")
    print(f"Evaluate the *best* checkpoint, not the final one: {output_dir}/best")


if __name__ == "__main__":
    main()
