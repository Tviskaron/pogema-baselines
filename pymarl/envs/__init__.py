from functools import partial
from pogema import pogema_v0

from utils.config_validation import ExperimentConfig


def env_fn(**kwargs):
    cfg = ExperimentConfig(**kwargs)
    cfg.env_args.grid_config.integration = 'PyMARL'
    return pogema_v0(grid_config=cfg.env_args.grid_config)


REGISTRY = {}

REGISTRY["pogema"] = partial(env_fn, integration='PyMARL')
