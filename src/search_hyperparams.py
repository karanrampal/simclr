#!/usr/bin/env python 3
"""Peform hyperparemeters search"""

import argparse
import itertools
import os
import sys
from subprocess import check_call

from utils import utils

PYTHON = sys.executable


def args_parser() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_dir",
        default="experiments/param_search",
        help="Directory containing params.json",
    )
    parser.add_argument(
        "--data_dir", default="./data", help="Directory containing the dataset"
    )
    return parser.parse_args()


def launch_training_job(
    model_dir: str, data_dir: str, job_name: str, params: utils.Params
) -> None:
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: Directory containing config, weights and log
        data_dir: Directory containing the dataset
        job_name: Name of the experiment to search hyperparameters
        params: Hyperparameters
    """
    model_dir = os.path.join(model_dir, job_name)
    os.makedirs(model_dir, exist_ok=True)

    json_path = os.path.join(model_dir, "params.json")
    params.save(json_path)

    cmd = f"{PYTHON} train.py --model_dir={model_dir} --data_dir {data_dir}"
    print(cmd)
    check_call(cmd, shell=True)


def main() -> None:
    """Main function"""
    args = args_parser()
    params = utils.Params(vars(args))

    configurations = {
        "learning_rate": [0.01, 0.001, 0.0001],
        "decay": [0.0, 0.01, 0.001],
    }
    conf_values = list(configurations.values())
    conf_names = list(configurations.keys())

    for vals in itertools.product(*conf_values):
        conf = dict(zip(conf_names, vals))
        params.update(conf)

        name = "_".join(str(key) + str(val) for key, val in conf.items())
        job_name = f"params_{name}"
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)


if __name__ == "__main__":
    main()
