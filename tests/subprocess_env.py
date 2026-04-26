import os


def so3lr_subprocess_env(extra=None):
    env = os.environ.copy()
    env.update({
        'JAX_TRACEBACK_FILTERING': 'off',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': '1',
        'TF_NUM_INTEROP_THREADS': '1',
        'WANDB_MODE': 'disabled',
    })
    if extra:
        env.update(extra)
    return env
