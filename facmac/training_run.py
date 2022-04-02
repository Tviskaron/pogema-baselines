import argparse
import json

import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy

import wandb
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from utils.config_validation import ExperimentConfig
from utils.logging import get_logger
import yaml

from training import run

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def get_flat_config(exp: ExperimentConfig):
    flat_cfg = {}
    for key, value in exp.dict().items():
        if key == 'env_args':
            flat_cfg.update(env_args=value)
        elif key == 'algo_settings':
            flat_cfg.update(**value[exp.algo])
        elif key == 'algo':
            continue
        else:
            flat_cfg.update(**value)

    return flat_cfg


def go_run(exp, params):
    flat_cfg = get_flat_config(exp)

    ex.add_config(flat_cfg)

    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    if exp.logging.use_wandb:
        wandb.init(project=exp.logging.project_name, config=exp.dict(), save_code=False, sync_tensorboard=True)

    ex.run_commandline(params)


def main():
    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store", default="config.yaml",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json config', required=False)

    params = parser.parse_args()
    if params.raw_config:
        config = json.loads(params.raw_config)
    else:
        with open(params.config_path, "r") as f:
            config = yaml.safe_load(f)
    exp = ExperimentConfig(**config)
    go_run(exp, sys.argv[:1])


if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     params = deepcopy(sys.argv)
#
#     # Get the defaults from default.yaml
#     with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
#         try:
#             config_dict = yaml.load(f)
#         except yaml.YAMLError as exc:
#             assert False, "default.yaml error: {}".format(exc)
#
#     # Load algorithm and env base configs
#     env_config = _get_config(params, "--env-config", "envs")
#     alg_config = _get_config(params, "--config", "algs")
#     # config_dict = {**config_dict, **env_config, **alg_config}
#     config_dict = recursive_dict_update(config_dict, env_config)
#     config_dict = recursive_dict_update(config_dict, alg_config)
#
#     # now add all the config to sacred
#     ex.add_config(config_dict)
#
#     # Save to disk by default for sacred
#     logger.info("Saving to FileStorageObserver in results/sacred.")
#     file_obs_path = os.path.join(results_path, "sacred")
#     ex.observers.append(FileStorageObserver.create(file_obs_path))
#
#     ex.run_commandline(params)
