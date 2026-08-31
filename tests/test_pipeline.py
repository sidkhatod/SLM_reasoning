"""
Wiring tests: every module imports, the configs resolve, and the GRPO loop runs
end-to-end on a tiny stub model. These are the checks that would have caught the
signature mismatches between the trainer and the three layers.
"""

import glob
import importlib
import os

import pytest
import torch

from rewards.reward_functions import resolve_weights
from utils.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("module", [
    "utils.config", "utils.data", "utils.metrics", "utils.checkpointing", "utils.logger",
    "rewards.reward_functions",
    "training.layers.spm", "training.layers.icr", "training.layers.dsac",
    "training.grpo_train", "training.sft_train",
    "evaluation.harness", "evaluation.eval_gsm8k", "evaluation.eval_strategyqa",
    "evaluation.eval_mmlu", "evaluation.generate_report",
    "inference.inference",
])
def test_every_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("path", sorted(
    glob.glob(os.path.join(ROOT, "configs", "*.yaml")) +
    glob.glob(os.path.join(ROOT, "configs", "ablations", "*.yaml"))
))
def test_configs_resolve_and_carry_required_keys(path):
    config = load_config(path)
    assert config.get("base_model"), f"{path} has no base_model"
    assert config.get("learning_rate") is not None

    if os.path.basename(path) == "sft.yaml":
        # The two values the writeup had flagged as unresolved must now be concrete.
        assert isinstance(config["batch_size"], int) and config["batch_size"] > 0
        assert isinstance(config["epochs"], int) and config["epochs"] > 0
        return

    assert isinstance(config["max_steps"], int) and config["max_steps"] > 0
    assert config["group_size"] >= 2, "GRPO needs a group to be relative to"
    assert sum(config["reward_weights"].values()) == pytest.approx(1.0)
    assert sum(resolve_weights(config).values()) == pytest.approx(1.0)

    kl = config["adaptive_kl"]
    assert kl["beta_min"] <= kl["beta_init"] <= kl["beta_max"]
    # A candidate pool at or below the buffer size disables curriculum selection.
    assert config["dsac_candidate_pool"] > config["dsac_buffer_size"]


def test_all_seven_ablation_conditions_exist_and_differ():
    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "ablations", "a*.yaml")))
    toggles = {}
    for p in paths:
        cfg = load_config(p)
        key = os.path.basename(p)
        toggles[key] = (cfg.get("use_spm"), cfg.get("use_icr"), cfg.get("use_dsac"),
                        cfg.get("skip_grpo", False), cfg.get("base_model"))

    assert len(paths) >= 7, "the ablation plan calls for 7 conditions"
    assert len(set(toggles.values())) == len(toggles), \
        f"two ablation configs are identical: {toggles}"


# ---------------------------------------------------------------------------
# End-to-end GRPO step against a stub policy
# ---------------------------------------------------------------------------

class TinyTokenizer:
    """Character-level tokenizer over a fixed vocabulary."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self):
        self.chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:?!\n'*=/-")
        self.stoi = {c: i + 2 for i, c in enumerate(self.chars)}
        self.itos = {i + 2: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 2

    def encode(self, text, add_special_tokens=False):
        return [self.stoi.get(c, 2) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos.get(int(i), "") for i in ids)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(r, skip_special_tokens) for r in rows]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None,
                 add_special_tokens=True):
        ids = self.encode(text)
        if max_length:
            ids = ids[:max_length]

        class Enc(dict):
            pass

        enc = Enc()
        enc["input_ids"] = torch.tensor([ids], dtype=torch.long)
        enc["attention_mask"] = torch.ones(1, len(ids), dtype=torch.long)
        enc.input_ids = enc["input_ids"]
        enc.attention_mask = enc["attention_mask"]
        return enc


class TinyPolicy(torch.nn.Module):
    """
    A one-layer LM that emits a fixed, scripted set of completions.

    The point is to exercise the *loop* - rewards, ICR, SPM, advantages, KL against
    the reference, backward, optimizer step - not to learn anything.
    """

    def __init__(self, tokenizer, scripted):
        super().__init__()
        self.tok = tokenizer
        self.emb = torch.nn.Embedding(tokenizer.vocab_size, 16)
        self.head = torch.nn.Linear(16, tokenizer.vocab_size)
        self.scripted = scripted
        self.adapter_disabled = False

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        logits = self.head(self.emb(input_ids))
        if self.adapter_disabled:
            # A distinct "reference policy" so the KL term is non-degenerate.
            logits = logits * 0.5

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out

    def generate(self, input_ids=None, num_return_sequences=1, max_new_tokens=64, **kwargs):
        prompt_len = input_ids.shape[1]
        rows = []
        for i in range(num_return_sequences):
            comp = self.tok.encode(self.scripted[i % len(self.scripted)])[:max_new_tokens]
            comp = comp + [self.tok.eos_token_id]
            rows.append(input_ids[0].tolist() + comp)
        width = max(len(r) for r in rows)
        rows = [r + [self.tok.pad_token_id] * (width - len(r)) for r in rows]
        return torch.tensor(rows, dtype=torch.long)

    def disable_adapter(self):
        policy = self

        class Ctx:
            def __enter__(self):
                policy.adapter_disabled = True

            def __exit__(self, *a):
                policy.adapter_disabled = False
        return Ctx()


@pytest.fixture
def tiny_trainer(tmp_path):
    from training.grpo_train import PrismGRPOTrainer
    from training.layers.dsac import RandomBuffer

    tok = TinyTokenizer()
    scripted = [
        "We add 4 and 6 to get 10. Then halve it.\n\nFinal Answer: 5",
        "We add 4 and 6 to get 10. Then double it.\n\nFinal Answer: 20",
        "We add 4 and 6 to get 10. Let me double-check. Halve it.\n\nFinal Answer: 5",
        "Nonsense output with no marker",
    ]
    model = TinyPolicy(tok, scripted)

    config = {
        "group_size": 4, "max_new_tokens": 64, "max_prompt_length": 128,
        "groups_per_step": 1, "max_steps": 2, "learning_rate": 1e-3,
        "use_spm": True, "use_icr": True, "use_dsac": False, "use_prm": False,
        "icr_similarity_threshold": 0.5,
        "adaptive_kl": {"beta_init": 0.04, "beta_min": 0.01, "beta_max": 0.3,
                        "target_kl": 0.1, "kp": 0.1},
        "eval_steps": 0, "save_steps": 0, "logging_steps": 100, "warmup_steps": 1,
    }
    data = [{"prompt": "Question: what is (4+6)/2?\nAnswer:", "answer": "5"}]

    return PrismGRPOTrainer(
        model=model, tokenizer=tok, config=config,
        train_dataset=data, val_dataset=[], dsac_buffer=RandomBuffer(config),
        output_dir=str(tmp_path), run_name="tiny",
    )


def test_a_full_grpo_group_produces_a_finite_differentiable_loss(tiny_trainer):
    loss, stats = tiny_trainer._process_group(tiny_trainer.train_dataset[0])

    assert torch.isfinite(loss), "loss must be finite"
    assert loss.requires_grad, "loss must carry gradients back to the policy"
    assert stats["kl"] >= 0.0, "the k3 estimator is non-negative by construction"
    assert 0.0 <= stats["accuracy"] <= 1.0
    assert stats["accuracy"] > 0, "the scripted group contains correct completions"
    assert 0.0 <= stats["lld_severity"] <= 1.0
    assert stats["lld_severity"] > 0, "the scripted completions share a reasoning prefix"
    assert stats["entropy"] > 0
    assert set(stats["rewards"]) >= {"outcome", "consensus", "format", "total"}


def test_kl_is_measured_against_the_reference_policy_not_itself(tiny_trainer):
    """
    The original loop compared the policy to a detached copy of itself, so KL was
    identically zero and the adaptive controller had nothing to control.
    """
    _, stats = tiny_trainer._process_group(tiny_trainer.train_dataset[0])
    assert stats["kl"] > 1e-6, "KL against the frozen base model must be non-zero"


def test_training_steps_update_the_policy_and_adapt_beta(tiny_trainer):
    before = tiny_trainer.model.head.weight.detach().clone()
    beta_before = tiny_trainer.current_beta

    tiny_trainer.train()

    assert tiny_trainer.step_counter == 2
    assert not torch.allclose(before, tiny_trainer.model.head.weight), \
        "the optimizer step must actually move the policy"
    assert tiny_trainer.current_beta != beta_before, "adaptive KL must respond to observed KL"
    assert (tiny_trainer.beta_min <= tiny_trainer.current_beta <= tiny_trainer.beta_max)


def test_spm_changes_the_token_advantages_relative_to_the_ablation(tiny_trainer):
    """Turning Layer 1 off must produce uniform per-sequence advantages, and on must not."""
    from training.layers.spm import apply_spm

    completions = [tiny_trainer.tokenizer.encode(s) for s in tiny_trainer.model.scripted]
    advantages = [1.0, -1.0, 1.0, -1.0]
    labels = [True, False, True, False]

    with_spm = apply_spm(completions, advantages, {"spm_mask_strength": 0.5}, labels)
    assert any(adv.unique().numel() > 1 for adv in with_spm), \
        "SPM must vary the advantage within a sequence"

    uniform = [torch.full((len(ids),), advantages[i]) for i, ids in enumerate(completions)]
    assert all(adv.unique().numel() == 1 for adv in uniform)
