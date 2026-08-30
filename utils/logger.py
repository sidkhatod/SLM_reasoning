import os
import sys

try:
    import wandb
except ImportError:  # wandb is optional - runs fall back to stdout logging
    wandb = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


def setup_wandb(config, default_project="prism-grpo", run_name=None):
    """
    Initializes Weights & Biases if it is installed and an API key is present.

    Returns the wandb module on success and None otherwise, so callers can log
    unconditionally through `utils.metrics` without guarding every call site.
    """
    load_dotenv()

    if wandb is None:
        print("wandb is not installed - metrics will print to stdout only.")
        return None

    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found in the environment (.env) - "
              "metrics will print to stdout only.")
        return None

    wandb.init(
        project=(config or {}).get("wandb_project", default_project),
        name=run_name,
        config=config,
    )
    return wandb


def configure_console():
    """
    Forces stdout/stderr to UTF-8 with replacement.

    Model completions routinely contain characters outside the Windows console's
    default cp1252 codepage (math symbols, curly quotes, CJK), and printing one
    raises UnicodeEncodeError mid-run. Replacing unencodable characters keeps a
    long training or inference session alive.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
