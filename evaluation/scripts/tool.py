#!/usr/bin/env python3
""" Helps run evaluation build. """

import argparse
import copy
import functools
import logging
import multiprocessing as mp
import os
import subprocess
from typing import Dict

import colorlog
import numpy as np
import toml

np.random.seed(42)

# --------------------------------------------------------------------------


def get_logger(name: str, log_file: str = "", console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if log_file:
        if os.path.isfile(log_file):
            os.remove(log_file)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s[%(asctime)s][%(name)s][%(levelname)s]%(reset)s"
                + " %(message_log_color)s%(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
                secondary_log_colors={"message": {"ERROR": "red", "CRITICAL": "red"}},
            )
        )
        logger.addHandler(ch)

    return logger


logger = get_logger("deacon")


# -------------------------------------------------------------------------


def generate_data_array(N: int) -> np.ndarray:
    return np.random.uniform(low=-2.1, high=2.1, size=N)


def generate_data_layer(name: str, cfg: Dict, data_file: str):

    if cfg["TYPE"] == "DEPTHWISE_SEPARABLE":
        # Create data for both depthwise and pointwise:
        with open(data_file, "a") as f:
            # depthwise
            f.write(f"BEGIN {name}_dw\n")
            N = cfg["K"] * cfg["K"] * cfg["C"]
            f.write(f"{N}\n")
            arr = generate_data_array(N)
            for x in arr:
                f.write(f"{x}\n")

            # pointwise
            f.write(f"BEGIN {name}_pw\n")
            N = cfg["F"] * cfg["C"]
            f.write(f"{N}\n")
            arr = generate_data_array(N)
            for x in arr:
                f.write(f"{x}\n")
    else:
        raise NotImplementedError(f'Type {cfg["TYPE"]} not recognized.')


def generate_data(cfg: Dict, root_dir: str, key: str) -> str:
    """Generate coefficient data."""

    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    data_file = os.path.join(data_dir, f"data-{key}.txt")
    if os.path.isfile(data_file):
        os.remove(data_file)

    logger.info(f"Data generated to file: {data_file}")
    if "NUM_LAYER" not in cfg:
        generate_data_layer(cfg["NAME"], cfg, data_file)
    else:
        for i in range(cfg["NUM_LAYER"]):
            generate_data_layer(cfg["NAME"] + str(i), cfg, data_file)

    return data_file


def runsim(args):
    """Run simulation."""

    logger.info(args)

    cfg = toml.load(args.cfg)
    logger.info(f"Loaded config: {cfg}")

    key = os.path.basename(args.cfg).split(".")[0]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.cfg)))
    logger.info(f"Root directory: {root_dir}")

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log_file = os.path.join(log_dir, f"{key}.runsim.stdout.log")
    stderr_log_file = os.path.join(log_dir, f"{key}.runsim.stderr.log")
    logger.info(f"Log file (stdout): {stdout_log_file}")
    logger.info(f"Log file (stderr): {stderr_log_file}")

    # Generate data file.
    data_file = generate_data(cfg, root_dir, key=key)

    cli_options = f"-n 2 -f {data_file}"

    cmd = ["make", "runsim"]
    for key, value in cfg.items():
        cmd.append(f"{key}={value}")
    cmd.append(f"COEFF_FILE={data_file}")
    cmd.append(f"CLI_OPTIONS={cli_options}")

    if args.dbg:
        cmd.append(f"DEBUG=true")

    logger.info(f"Command to run: {cmd}")
    subprocess.run(
        cmd,
        cwd=root_dir,
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )


def build(args):
    """Run simulation."""

    logger.info(args)

    cfg = toml.load(args.cfg)
    logger.info(f"Loaded config: {cfg}")

    key = os.path.basename(args.cfg).split(".")[0]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.cfg)))
    logger.info(f"Root directory: {root_dir}")

    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log_file = os.path.join(log_dir, f"{key}.build.stdout.log")
    stderr_log_file = os.path.join(log_dir, f"{key}.build.stderr.log")
    logger.info(f"Log file (stdout): {stdout_log_file}")
    logger.info(f"Log file (stderr): {stderr_log_file}")

    # Generate data file.
    data_file = generate_data(cfg, root_dir, key=key)

    cmd = ["make", "build"]
    for key, value in cfg.items():
        cmd.append(f"{key}={value}")
    cmd.append(f"COEFF_FILE={data_file}")

    logger.info(f"Command to run: {cmd}")
    subprocess.run(
        cmd,
        cwd=root_dir,
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )


# -------------------------------------------------------------------------


def process(args):
    args.cfg = args.cfg[args.index]  # overwrite the list
    args.func(args)


def main():
    parser = argparse.ArgumentParser(description="Deacon tool.")
    parser.add_argument("--cfg", nargs="+", default=[], help="Configuration files")
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of parallelised jobs"
    )
    parser.add_argument("-i", "--index", type=int, default=0, help="Job index")

    subparsers = parser.add_subparsers()

    runsim_parser = subparsers.add_parser("runsim", help="Run simulation")
    runsim_parser.add_argument("--dbg", action="store_true", help="Debug flag")
    runsim_parser.set_defaults(func=runsim)

    build_parser = subparsers.add_parser("build", help="Run build")
    build_parser.set_defaults(func=build)

    args = parser.parse_args()

    # Prepare parallel runs
    args_list = [copy.deepcopy(args) for i in range(len(args.cfg))]
    for i, args in enumerate(args_list):
        args.index = i

    with mp.Pool(args.jobs) as p:
        p.map(process, args_list)


if __name__ == "__main__":
    main()
