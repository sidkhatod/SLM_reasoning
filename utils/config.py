import os

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins on conflicts."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path, _seen=None):
    """
    Loads a YAML config, resolving an optional `extends:` key.

    `extends` names a parent config (relative to the child's directory, or to the repo
    root) whose keys are deep-merged underneath the child's. The ablation configs use
    it so all seven conditions share one set of hyperparameters and differ only in the
    layer toggles they override - a hyperparameter drifting between conditions would
    invalidate the comparison the ablation exists to make.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _seen = _seen or set()
    if config_path in _seen:
        raise ValueError(f"Circular `extends` chain involving {config_path}")
    _seen.add(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent_ref = config.pop("extends", None)
    if not parent_ref:
        return config

    candidates = [
        os.path.join(os.path.dirname(config_path), parent_ref),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(config_path))), parent_ref),
        parent_ref,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return _deep_merge(load_config(candidate, _seen), config)

    raise FileNotFoundError(
        f"`extends: {parent_ref}` in {config_path} could not be resolved."
    )
