import argparse

import json
import os
import subprocess
import time

import yaml

from utils.config_validation import ExperimentConfig

from utils.gs2dict import generate_variants


def main():
    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store", default="configs/8x8.yaml",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json config', required=False)

    params = parser.parse_args()
    if params.raw_config:
        config = json.loads(params.raw_config)
    else:
        with open(params.config_path, "r") as f:
            config = yaml.safe_load(f)

    for resolved_vars, spec in generate_variants(config):
        exp = ExperimentConfig(**spec)

        cmd = f"python3 training_run.py --raw_config '{json.dumps(exp.dict())}'"
        env_vars = os.environ.copy()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=env_vars)
        output, err = process.communicate()
        print(output, err)
        exit_code = process.wait()

        if exit_code != 0:
            break

        time.sleep(5)


if __name__ == '__main__':
    main()