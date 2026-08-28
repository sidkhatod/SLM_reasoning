import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    get_scheduler,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import wandb

import sys
# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import setup_wandb
from utils.checkpointing import save_checkpoint, load_checkpoint


def format_example_template(prompt: str, cot: str, answer: str):
    """
    Consistent prompt -> CoT -> answer template.
    Reusable for GRPO phase later.
    """
    return {
        "prompt": f"Question: {prompt}\n\nAnswer: Let's think step by step.\n",
        "completion": f"{cot}\n\nFinal Answer: {answer}"
    }


def prepare_datasets(tokenizer, max_length):
    print("Loading datasets...")
    
    def format_and_tokenize(prompt, cot, answer):
        parts = format_example_template(prompt, cot, answer)
        text = parts["prompt"] + parts["completion"] + tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=max_length)

    # 1. GSM8K
    try:
        ds_gsm8k = load_dataset("gsm8k", "main", split="train")
        def map_gsm8k(x):
            parts = x["answer"].split("####")
            cot = parts[0].strip()
            ans = parts[1].strip() if len(parts) > 1 else ""
            return format_and_tokenize(x["question"], cot, ans)
        ds_gsm8k = ds_gsm8k.map(map_gsm8k, remove_columns=ds_gsm8k.column_names)
    except Exception as e:
        print(f"Failed to load GSM8K: {e}")
        ds_gsm8k = None

    # 2. AQuA-RAT
    try:
        ds_aqua = load_dataset("aqua_rat", "raw", split="train")
        def map_aqua(x):
            opts = "\n".join(x["options"])
            prompt = f"{x['question']}\nOptions:\n{opts}"
            return format_and_tokenize(prompt, x["rationale"], x["correct"])
        ds_aqua = ds_aqua.map(map_aqua, remove_columns=ds_aqua.column_names)
    except Exception as e:
        print(f"Failed to load AQuA-RAT: {e}")
        ds_aqua = None

    # 3. StrategyQA
    try:
        ds_strategy = load_dataset("tau/strategyqa", split="train")
        def map_strategy(x):
            cot = ""
            if "facts" in x and isinstance(x["facts"], list):
                cot = "Facts:\n" + "\n".join(f"- {f}" for f in x["facts"])
            ans = str(x.get("answer", ""))
            return format_and_tokenize(x["question"], cot, ans)
        ds_strategy = ds_strategy.map(map_strategy, remove_columns=ds_strategy.column_names)
    except Exception as e:
        print(f"Failed to load StrategyQA: {e}")
        ds_strategy = None
        
    datasets_to_concat = [ds for ds in [ds_gsm8k, ds_aqua, ds_strategy] if ds is not None]
    if not datasets_to_concat:
        raise ValueError("No datasets could be loaded! Please check dataset names or internet connection.")
        
    train_dataset = concatenate_datasets(datasets_to_concat)
    train_dataset = train_dataset.shuffle(seed=42)
    
    # We remove string columns since PyTorch DataLoader doesn't handle them naturally
    # DataCollatorForLanguageModeling only needs input_ids and attention_mask
    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in train_dataset.column_names if c not in cols_to_keep]
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    
    train_dataset.set_format(type="torch")
    return train_dataset


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen2.5-7B")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size override")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps override")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_qwen2.5-7b/", help="Output directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Setup WandB
    setup_wandb(config, default_project="prism-grpo", run_name="sft-run")
    
    # Setup tokenizer
    model_id = config.get("base_model", "Qwen/Qwen2.5-7B")
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    max_length = config.get("max_seq_length", 512)
    train_dataset = prepare_datasets(tokenizer, max_length)
    print(f"Combined dataset size: {len(train_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )

    # Model configuration for 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading base model {model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        
    model = prepare_model_for_kbit_training(model)
    
    # Extract LoRA target modules
    target_modules = config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    lora_config = LoraConfig(
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optimizer & Scheduler
    lr = float(config.get("learning_rate", 2e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    epochs = config.get("epochs")
    if epochs is None: 
        epochs = 1
        
    total_steps = (len(train_dataloader) // args.grad_accum_steps) * epochs
    
    lr_schedule = config.get("lr_schedule", "cosine")
    scheduler = get_scheduler(
        name=lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if provided
    start_step = 0
    if args.resume_from:
        start_step, extra_state = load_checkpoint(model, optimizer, scheduler, args.resume_from)

    model.train()
    
    step_count = start_step
    running_loss = 0.0
    optimizer.zero_grad()
    
    print("Starting training...")
    
    progress_bar = tqdm(total=total_steps, initial=start_step)
    
    global_batch_idx = 0
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # Skip if resuming
            if global_batch_idx < start_step * args.grad_accum_steps:
                global_batch_idx += 1
                continue
                
            # Move to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False # Important for gradient checkpointing
            )
            
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item()
            
            global_batch_idx += 1
            
            if global_batch_idx % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                step_count += 1
                progress_bar.update(1)
                
                # Logging
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "loss": running_loss,
                    "learning_rate": current_lr,
                    "step": step_count
                })
                
                # Checkpointing and Memory Print
                if step_count % args.save_steps == 0:
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(f"\n[Step {step_count}] GPU Memory: Allocated {mem_allocated:.2f} GB, Reserved {mem_reserved:.2f} GB\n")
                    
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step_count}")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step_count,
                        extra_state={"epoch": epoch},
                        path=ckpt_path
                    )
                
                running_loss = 0.0

    progress_bar.close()
    
    # Save final model
    print(f"Training complete. Saving final LoRA adapter to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
