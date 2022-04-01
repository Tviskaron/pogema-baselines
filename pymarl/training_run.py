import argparse
import collections
import json
import os
import sys
from copy import deepcopy
from os.path import dirname, abspath

import numpy as np
import torch as th
import yaml
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from pymarl.utils.config_validation import ExperimentConfig
from training import run
from utils.logging import get_logger

import wandb

# SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(abspath(__file__)), "results")

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
        with open(os.path.join(os.path.dirname(__file__), "configs", subfolder, "{}.yaml".format(config_name)),
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
    if exp.algo == 'COMA':
        exp.env_args.state_last_action = False

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
    parser = argparse.ArgumentParser(description='Process training configs.')

    parser.add_argument('--config_path', type=str, action="store", default="configs.yaml",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json configs', required=False)

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
