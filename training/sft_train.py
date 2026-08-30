"""
Phase 1: SFT warmup.

Teaches the model the `<reasoning> ... Final Answer: <value>` shape that every
downstream component depends on - the GRPO format/consistency rewards, the ICR step
splitter, and the evaluation answer extractor all parse this template. Running GRPO
straight off the base model is what produces the cold-start instability the warmup
exists to avoid.

Loss is computed on the completion tokens only: the prompt is context, and training
on it wastes capacity teaching the model to generate questions.

    python training/sft_train.py --config configs/sft.yaml
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.data import build_completion, build_prompt, load_reasoning_dataset
from utils.logger import configure_console, setup_wandb

try:
    import wandb
except ImportError:
    wandb = None

IGNORE_INDEX = -100


def format_example_template(prompt: str, cot: str, answer: str):
    """
    Consistent prompt -> CoT -> answer template, shared with the GRPO phase and the
    evaluation harness via `utils.data`.
    """
    return {"prompt": build_prompt(prompt), "completion": build_completion(cot, answer)}


def prepare_datasets(tokenizer, max_length, dataset_names, limit=None):
    """Tokenizes the aggregated CoT corpus with completion-only labels."""
    raw = load_reasoning_dataset(names=dataset_names, split="train")
    if limit:
        raw = raw[:limit]

    examples = []
    skipped = 0
    for item in raw:
        if not item["cot"].strip():
            skipped += 1
            continue

        parts = format_example_template(item["question"], item["cot"], item["answer"])
        prompt_ids = tokenizer(parts["prompt"], add_special_tokens=True).input_ids
        completion_ids = tokenizer(
            parts["completion"] + tokenizer.eos_token, add_special_tokens=False
        ).input_ids

        input_ids = (prompt_ids + completion_ids)[:max_length]
        # Mask the prompt out of the loss.
        labels = ([IGNORE_INDEX] * len(prompt_ids) + completion_ids)[:max_length]

        # A truncated example whose completion was cut away entirely carries no signal.
        if all(l == IGNORE_INDEX for l in labels):
            skipped += 1
            continue

        examples.append({"input_ids": input_ids, "labels": labels})

    print(f"Prepared {len(examples)} SFT examples ({skipped} skipped: no CoT trace "
          "or truncated past the completion).")
    return examples


def make_collator(pad_token_id):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention_mask = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
            attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="SFT warmup for PRISM-GRPO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                        help="Override config gradient_accumulation_steps")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--save_steps", type=int, default=None, help="Save every N optimizer steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint dir to resume from")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset_limit", type=int, default=None, help="Cap examples (smoke tests)")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    batch_size = args.batch_size or int(config.get("batch_size", 4))
    grad_accum = args.grad_accum_steps or int(config.get("gradient_accumulation_steps", 8))
    epochs = args.epochs or int(config.get("epochs", 2))
    save_steps = args.save_steps or int(config.get("save_steps", 200))
    output_dir = args.output_dir or config.get("output_dir", "outputs/sft_qwen2.5-7b")

    if not args.no_wandb:
        setup_wandb(config, default_project="prism-grpo", run_name="sft-warmup")

    model_id = config.get("base_model", "Qwen/Qwen2.5-7B")
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(config.get("max_seq_length", 512))
    train_examples = prepare_datasets(
        tokenizer, max_length,
        config.get("datasets", ["gsm8k", "aqua_rat", "strategyqa"]),
        limit=args.dataset_limit,
    )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=make_collator(tokenizer.pad_token_id),
    )

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if config.get("quantization", "4bit") == "4bit" and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("CUDA unavailable - loading in fp32 without 4-bit quantization.")
        load_kwargs["torch_dtype"] = torch.float32

    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    use_gc = bool(config.get("gradient_checkpointing", True))
    if "quantization_config" in load_kwargs:
        # prepare_model_for_kbit_training enables gradient checkpointing *and* the
        # input-require-grads hook it needs; calling enable() beforehand does not.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    elif use_gc:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
        target_modules=config.get("lora_target_modules",
                                  ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    lr = float(config.get("learning_rate", 2e-4))
    # Only the LoRA parameters carry gradients; handing the frozen 4-bit base weights
    # to AdamW would allocate optimizer state for them.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    steps_per_epoch = max(1, len(train_dataloader) // grad_accum)
    total_steps = steps_per_epoch * epochs

    scheduler = get_scheduler(
        name=config.get("lr_schedule", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=int(config.get("warmup_steps", 100)),
        num_training_steps=total_steps,
    )

    start_step = 0
    if args.resume_from:
        start_step, _ = load_checkpoint(model, optimizer, scheduler, args.resume_from)

    model.train()
    print(f"Training: {epochs} epochs x {steps_per_epoch} steps "
          f"(batch {batch_size} x accum {grad_accum} = effective {batch_size * grad_accum})")

    step_count = start_step
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(total=total_steps, initial=start_step)

    micro_step = 0
    skip_until = start_step * grad_accum
    done = False

    for epoch in range(epochs):
        if done:
            break
        for batch in train_dataloader:
            if micro_step < skip_until:
                micro_step += 1
                continue

            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch, use_cache=False)

            loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += float(loss.item())
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step_count += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{running_loss:.4f}")

                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log({"loss": running_loss,
                               "learning_rate": scheduler.get_last_lr()[0],
                               "step": step_count})

                if save_steps > 0 and step_count % save_steps == 0:
                    if torch.cuda.is_available():
                        print(f"\n[Step {step_count}] GPU memory: "
                              f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated, "
                              f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB reserved\n")
                    save_checkpoint(model, optimizer, scheduler, step_count,
                                    {"epoch": epoch}, os.path.join(output_dir, f"checkpoint-{step_count}"))

                running_loss = 0.0

                if step_count >= total_steps:
                    done = True
                    break

    progress_bar.close()

    print(f"Training complete. Saving final LoRA adapter to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Next: python training/grpo_train.py --config configs/grpo_base.yaml "
          f"--sft_checkpoint {output_dir}")


if __name__ == "__main__":
    main()
