import wandb
import os
from dotenv import load_dotenv

def setup_wandb(config, default_project="prism-grpo", run_name=None):
    """
    Initialize Weights & Biases with the given configuration.
    """
    load_dotenv()
    
    # Check if WANDB_API_KEY is available
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found in environment variables. Weights & Biases logging might fail.")
    
    # Project name can be overridden by config
    project_name = config.get("wandb_project", default_project)
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )
    return wandb
