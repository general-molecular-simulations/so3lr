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
    init_args = dict(training.get('wandb_init_args') or {})
    # Log the run configuration, but not the W&B client arguments themselves
    # (they may carry entity/project settings or credentials).
    logged = {**config, 'training': {k: v for k, v in training.items()
                                     if k != 'wandb_init_args'}}
    wandb.init(config=logged, **init_args)
    return True
