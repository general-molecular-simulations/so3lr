"""Opt-in Weights & Biases integration."""


def initialize_wandb(config) -> bool:
    """Call ``wandb.init`` only if ``training.use_wandb`` is literally true.

    Returns the flag for the training loop. wandb is imported lazily, so the
    optional client is only needed when W&B is enabled.
    """
    config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    training = config.get('training', {})
    if training.get('use_wandb', False) is not True:
        return False
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "training.use_wandb is true but wandb is not installed; install "
            "the optional extra with `pip install 'so3lr[wandb]'`."
        ) from exc
    wandb.init(config=config, **(training.get('wandb_init_args') or {}))
    return True
