import os
import torch

def save_checkpoint(model, optimizer, scheduler, step, extra_state, path):
    """
    Saves a checkpoint suitable for resuming training on platforms like Colab/Kaggle.
    
    Args:
        model: The model (expected to be a PeftModel)
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        step (int): Current training step
        extra_state (dict): Any additional state to save (e.g., random states, epochs)
        path (str): Directory path to save the checkpoint
    """
    os.makedirs(path, exist_ok=True)
    
    # Save the LoRA adapter weights
    # Assuming model is a PEFT model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        # Fallback if not a standard PEFT model
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        
    # Save optimizer and scheduler states
    train_state = {
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'step': step,
        'extra_state': extra_state
    }
    
    torch.save(train_state, os.path.join(path, "train_state.pt"))
    print(f"Checkpoint saved at step {step} to {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """
    Restores the model adapter weights, optimizer, and scheduler states.
    
    Args:
        model: The PEFT model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        path (str): Directory path of the saved checkpoint
        
    Returns:
        int: The step to resume training from. Returns 0 if checkpoint not found.
        dict: The extra_state dictionary saved previously, or empty dict.
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return 0, {}
        
    print(f"Loading checkpoint from {path}...")
    
    # Load LoRA weights
    try:
        from peft import set_peft_model_state_dict
        adapter_path_bin = os.path.join(path, "adapter_model.bin")
        adapter_path_safetensors = os.path.join(path, "adapter_model.safetensors")
        model_path_pt = os.path.join(path, "model.pt")
        
        if os.path.exists(adapter_path_bin):
            adapters_weights = torch.load(adapter_path_bin, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        elif os.path.exists(adapter_path_safetensors):
            from safetensors.torch import load_file
            adapters_weights = load_file(adapter_path_safetensors)
            set_peft_model_state_dict(model, adapters_weights)
        elif os.path.exists(model_path_pt):
            model.load_state_dict(torch.load(model_path_pt, map_location="cpu"))
        else:
            print("Warning: Could not find model weights in the checkpoint directory.")
    except ImportError:
        print("Warning: PEFT not installed, unable to properly load adapter weights.")

    train_state_path = os.path.join(path, "train_state.pt")
    if os.path.exists(train_state_path):
        train_state = torch.load(train_state_path, map_location="cpu")
        
        if optimizer and train_state.get('optimizer'):
            optimizer.load_state_dict(train_state['optimizer'])
            
        if scheduler and train_state.get('scheduler'):
            scheduler.load_state_dict(train_state['scheduler'])
            
        step = train_state.get('step', 0)
        extra_state = train_state.get('extra_state', {})
        
        print(f"Successfully loaded checkpoint at step {step}.")
        return step, extra_state
    
    print("Warning: train_state.pt not found. Only model weights were potentially loaded.")
    return 0, {}
