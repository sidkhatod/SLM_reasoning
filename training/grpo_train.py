import argparse
import os
import sys
import yaml
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
import math
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.spm import apply_spm
from layers.icr import cluster_and_align_steps
from layers.dsac import CurriculumBuffer
from rewards.reward_functions import compute_total_reward
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.metrics import log_grpo_metrics
try:
    import wandb
except ImportError:
    wandb = None

# 1. Custom Iterable Dataset for DSAC
class CurriculumDataset(IterableDataset):
    def __init__(self, full_dataset, buffer):
        self.full_dataset = full_dataset
        self.buffer = buffer
        
    def __iter__(self):
        while True:
            # Yield dynamically sampled problem
            yield self.buffer.sample_problem(self.full_dataset)

# 2. Custom GRPO Trainer subclass
class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, dsac_buffer=None, dsac_config=None, target_kl=0.1, beta_init=0.04, beta_min=0.01, beta_max=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.dsac_buffer = dsac_buffer
        self.dsac_config = dsac_config
        self.target_kl = target_kl
        
        # Adaptive KL state
        self.current_beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        self.step_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to inject SPM (token-level advantages) and ICR (group-level consensus).
        Since standard TRL GRPOTrainer only supports scalar advantages, we intercept the loss calculation.
        """
        # NOTE: This overrides the standard compute_loss to allow for our custom custom architectures.
        # In a real environment, you might need to adapt this to the specific TRL version's inputs.
        
        # 1. Get Prompts
        prompts = inputs.get("prompt", [])
        if isinstance(prompts, torch.Tensor):
            # Decode if they are already tokenized
            prompts = self.processing_class.batch_decode(prompts, skip_special_tokens=True)
            
        if not prompts:
            return (torch.tensor(0.0, device=model.device, requires_grad=True), {}) if return_outputs else torch.tensor(0.0, device=model.device, requires_grad=True)

        prompt = prompts[0] # Assumes batch size 1 for simplicity in GRPO prompt iteration
        group_size = self.args.num_generations if hasattr(self.args, "num_generations") else 8

        # 2. Generate G completions
        encoded_prompt = self.processing_class(prompt, return_tensors="pt").to(model.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **encoded_prompt,
                max_new_tokens=self.args.max_completion_length if hasattr(self.args, "max_completion_length") else 512,
                num_return_sequences=group_size,
                do_sample=True,
                temperature=0.9,
                pad_token_id=self.processing_class.eos_token_id
            )
        model.train()
        
        prompt_len = encoded_prompt.input_ids.shape[1]
        completion_ids = outputs[:, prompt_len:]
        completions_str = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Ground truth
        gt = inputs.get("answer", [""])[0] if "answer" in inputs else ""

        # 3. Apply ICR (Implicit Consensus Rewards)
        use_icr = self.dsac_config.get("use_icr", True)
        if use_icr:
            icr_rewards, icr_stats = cluster_and_align_steps(completions_str)
        else:
            icr_rewards = [0.0] * group_size
            icr_stats = {}
            if getattr(self, "_icr_warned", False) is False:
                print("\n[Ablation] ICR is disabled. Consensus reward is 0. Weight NOT redistributed unless set in config.")
                self._icr_warned = True
        group_context = {"icr_rewards": icr_rewards}
        
        # 4. Calculate total reward
        total_rewards, reward_components = compute_total_reward(prompt, completions_str, gt, group_context, self.dsac_config)
        rewards_tensor = torch.tensor(total_rewards, device=model.device, dtype=torch.float32)
        
        # Normalize rewards into scalar advantages
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward

        # 5. Apply SPM (Semantic Prefix Masking)
        use_spm = self.dsac_config.get("use_spm", True)
        if use_spm:
            correct_completions = [c for i, c in enumerate(completions_str) if total_rewards[i] > mean_reward.item()]
            incorrect_completions = [c for i, c in enumerate(completions_str) if total_rewards[i] <= mean_reward.item()]
            
            token_advantages, lld_severity = apply_spm(
                correct_completions, 
                incorrect_completions, 
                completion_ids, 
                advantages, 
                self.processing_class, 
                mask_strength=0.5
            )
        else:
            lld_severity = 0.0
            token_advantages = []
            for i, comp in enumerate(completions_str):
                seq_len = len(self.processing_class.encode(comp, add_special_tokens=False))
                token_advantages.append(torch.full((seq_len,), advantages[i].item(), dtype=torch.float32))
            if getattr(self, "_spm_warned", False) is False:
                print("\n[Ablation] SPM is disabled. Token advantages are uniform.")
                self._spm_warned = True

        # 6. Calculate Policy Loss and KL
        # Forward pass for log probs
        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        total_kl = 0.0
        
        for i in range(group_size):
            # Forward pass to get active log probs
            comp_ids = completion_ids[i:i+1]
            full_input_ids = torch.cat([encoded_prompt.input_ids, comp_ids], dim=1)
            
            logits = model(full_input_ids).logits
            comp_logits = logits[:, prompt_len-1:-1, :]
            
            # Simple log prob calculation
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            nll = loss_fct(comp_logits.squeeze(0), comp_ids.squeeze(0))
            
            # Assume old_log_prob is detached current log prob for this simplified loop 
            # (In full GRPO, we need a reference model. We simulate reference model loosely here or assume 1 step PPO)
            with torch.no_grad():
                old_nll = nll.detach()
                
            log_ratio = -nll - (-old_nll)
            ratio = torch.exp(log_ratio)
            
            # Use token_advantages[i] which applies SPM masking
            adv = token_advantages[i].to(model.device)
            if adv.shape[0] > nll.shape[0]:
                adv = adv[:nll.shape[0]]
            elif adv.shape[0] < nll.shape[0]:
                pad = torch.zeros(nll.shape[0] - adv.shape[0], device=model.device)
                adv = torch.cat([adv, pad])
                
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
            pg_loss = torch.max(pg_loss1, pg_loss2)
            
            # KL approximation
            kl = (nll - old_nll) 
            total_kl += kl.mean().item()
            
            loss = loss + (pg_loss.mean() + self.current_beta * kl.mean())
            
        loss = loss / group_size
        total_kl = total_kl / group_size

        # 7. Adaptive KL Update (PID)
        error = self.target_kl - total_kl
        kp = 0.1 # proportional gain
        self.current_beta = self.current_beta * math.exp(kp * error)
        self.current_beta = max(self.beta_min, min(self.beta_max, self.current_beta))

        # 8. Logging and DSAC Refresh
        self.step_counter += 1
        ivs_score = icr_stats.get("discriminative_clusters", 0)
        
        log_grpo_metrics(
            step=self.step_counter,
            lld_severity=lld_severity,
            ivs_score=ivs_score,
            beta=self.current_beta,
            kl=total_kl,
            rewards=reward_components
        )
        
        if self.step_counter % 50 == 0:
            self.dsac_buffer.refresh(model, self.processing_class, self.train_dataset.full_dataset, buffer_size=1000)
            
        if self.step_counter % 10 == 0:
            mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            print(f"Step {self.step_counter} | Loss: {loss.item():.4f} | KL: {total_kl:.4f} | Beta: {self.current_beta:.4f} | VRAM: {mem:.2f}GB")

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_base.yaml")
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    
    parser.add_argument("--use_spm", action="store_true", default=None)
    parser.add_argument("--no_spm", action="store_false", dest="use_spm")
    parser.add_argument("--use_icr", action="store_true", default=None)
    parser.add_argument("--no_icr", action="store_false", dest="use_icr")
    parser.add_argument("--use_dsac", action="store_true", default=None)
    parser.add_argument("--no_dsac", action="store_false", dest="use_dsac")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.group_size is not None:
        config["group_size"] = args.group_size
        
    if args.use_spm is not None:
        config["use_spm"] = args.use_spm
    if args.use_icr is not None:
        config["use_icr"] = args.use_icr
    if args.use_dsac is not None:
        config["use_dsac"] = args.use_dsac

    if args.run_name is None:
        args.run_name = os.path.splitext(os.path.basename(args.config))[0]

    print(f"Initializing GRPO Training: {args.run_name}")
    if wandb is not None:
        wandb.init(project="prism-grpo", name=args.run_name, config=config)

    # 1. Load Model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Load SFT weights
    if os.path.exists(os.path.join(args.sft_checkpoint, "adapter_model.bin")) or os.path.exists(os.path.join(args.sft_checkpoint, "adapter_model.safetensors")):
        model.load_adapter(args.sft_checkpoint, "default")

    # 2. Dataset
    # Load synthetic raw dataset (e.g. GSM8K placeholder)
    raw_dataset = load_dataset("gsm8k", "main", split="train")
    
    # Format Dataset for GRPO
    def format_fn(example):
        return {
            "prompt": f"Question: {example['question']}\nAnswer: Let's think step by step.",
            "answer": example['answer'].split('####')[-1].strip()
        }
    formatted_dataset = [format_fn(ex) for ex in raw_dataset]
    
    # 3. Setup DSAC Curriculum Buffer
    use_dsac = config.get("use_dsac", True)
    if use_dsac:
        dsac_buffer = CurriculumBuffer(config)
        # Initial fill
        dsac_buffer.refresh(model, tokenizer, formatted_dataset, buffer_size=1000, sample_size=100)
    else:
        print("[Ablation] DSAC is disabled. Using random sampling from full dataset.")
        class DummyBuffer:
            def sample_problem(self, full_dataset):
                import random
                return random.choice(full_dataset)
            def refresh(self, *args, **kwargs):
                pass
            @property
            def buffer(self):
                return []
            @buffer.setter
            def buffer(self, val):
                pass
        dsac_buffer = DummyBuffer()
        
    train_iterable = CurriculumDataset(formatted_dataset, dsac_buffer)

    # 4. Trainer
    training_args = GRPOConfig(
        output_dir=f"outputs/{args.run_name}",
        learning_rate=float(config.get("learning_rate", 1e-5)),
        max_steps=1000,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1,
        num_generations=config.get("group_size", 8),
        max_completion_length=config.get("max_new_tokens", 512),
        logging_steps=10,
        save_steps=100
    )
    
    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_iterable,
        processing_class=tokenizer,
        dsac_buffer=dsac_buffer,
        dsac_config=config,
        target_kl=0.1,
        beta_init=config["adaptive_kl"]["beta_init"],
        beta_min=config["adaptive_kl"]["beta_min"],
        beta_max=config["adaptive_kl"]["beta_max"]
    )

    # 5. Resume logic
    if args.resume_from:
        step, extra_state = load_checkpoint(model, trainer.optimizer, trainer.lr_scheduler, args.resume_from)
        if step > 0:
            trainer.step_counter = step
            trainer.current_beta = extra_state.get("beta", config["adaptive_kl"]["beta_init"])
            if "dsac_buffer" in extra_state:
                dsac_buffer.buffer = extra_state["dsac_buffer"]
            print(f"Resumed at step {step} with beta {trainer.current_beta:.4f}")

    # 6. Train
    print("Starting GRPO Training...")
    trainer.train()
    
    # 7. Final Save
    save_path = f"outputs/{args.run_name}_final"
    extra_state = {
        "beta": trainer.current_beta,
        "dsac_buffer": dsac_buffer.buffer
    }
    save_checkpoint(model, trainer.optimizer, trainer.lr_scheduler, trainer.step_counter, extra_state, save_path)
    print("Training Complete.")

if __name__ == "__main__":
    main()
