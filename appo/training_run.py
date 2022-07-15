import json
from argparse import Namespace
from pathlib import Path

import gym
import numpy as np
import yaml
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES

from sample_factory.utils.utils import log

import wandb
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

import sys
from models.residual_net import ResnetEncoder
from utils.config_validation import Experiment, Environment


def make_pogema(env_cfg: Environment = Environment()):
    from pogema.integrations.make_pogema import make_pogema
    if env_cfg.name == 'Pogema-v0':
        env = make_pogema(grid_config=env_cfg.grid_config, integration='SampleFactory')
    else:
        env = gym.make(env_cfg.name, integration='SampleFactory')
    return env


def create_pogema_env(full_env_name, cfg=None, env_config=None):
    environment_config: Environment = Environment(**cfg.full_config['environment'])
    if env_config is None or env_config.get("remove_seed", True):
        environment_config.grid_config.seed = None
    return make_pogema(environment_config)


def override_default_params_func(env, parser):
    parser.set_defaults(
        encoder_custom='pogema_residual',
        hidden_size=128,
    )


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='Pogema',
        make_env_func=create_pogema_env,
        override_default_params_func=override_default_params_func,
    )

    register_custom_encoder('pogema_residual', ResnetEncoder)

    EXTRA_EPISODIC_STATS_PROCESSING.append(pogema_extra_episodic_stats_processing)
    EXTRA_PER_POLICY_SUMMARIES.append(pogema_extra_summaries)


def pogema_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    pass


def pogema_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')


def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


def main():
    register_custom_components()

    import argparse

    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json config', required=False)

    parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                        help='Run wandb in thread mode. Usefull for some setups.', required=False)

    params = parser.parse_args()

    if params.raw_config:
        config = json.loads(params.raw_config)
    else:
        if params.config_path is None:
            raise ValueError("You should specify --config_path or --raw_config argument!")
        with open(params.config_path, "r") as f:
            config = yaml.safe_load(f)

    exp, flat_config = validate_config(config)
    log.debug(exp.global_settings.experiments_root)

    if exp.global_settings.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=exp.name, config=exp.dict(), save_code=False, sync_tensorboard=True, anonymous="allow")

    status = run_algorithm(flat_config)
    if exp.global_settings.use_wandb:
        import shutil
        path = Path(exp.global_settings.train_dir) / exp.global_settings.experiments_root
        zip_name = str(path)
        shutil.make_archive(zip_name, 'zip', path)
        wandb.save(zip_name + '.zip')
    return status


if __name__ == '__main__':
    sys.exit(main())
