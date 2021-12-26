#!/usr/bin/env python3
""" Helps run evaluation build. """

import argparse
import copy
import functools
import logging
import multiprocessing as mp
import os
import subprocess
from typing import Dict, List

import colorlog
import numpy as np
import toml

np.random.seed(42)

TMPL_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def get_num_inputs(cfg: Dict) -> List[str]:
    if "INPUT" not in cfg:
        return 1
    if not isinstance(cfg["INPUT"], list):
        return 1
    return len(cfg["INPUT"])


def generate_data_array(N: int) -> np.ndarray:
    return np.random.uniform(low=-2.1, high=2.1, size=N)


def generate_data_layer(name: str, cfg: Dict, data_file: str):

    if cfg["TYPE"] == "STANDARD":
        for i in range(get_num_inputs(cfg)):
            with open(data_file, "a") as f:
                f.write(f"BEGIN {name}_{i}\n")
                # HACK:
                C = cfg["C"]
                if i > 0 and "SHORTCUT_C" in cfg:
                    C = cfg["SHORTCUT_C"]
                N = cfg["K"] * cfg["K"] * C * cfg["F"]
                f.write(f"{N}\n")
                arr = generate_data_array(N)
                for x in arr:
                    f.write(f"{x}\n")

    elif cfg["TYPE"] == "POINTWISE":
        with open(data_file, "a") as f:
            f.write(f"BEGIN {name}\n")
            N = cfg["C"] * cfg["F"]
            f.write(f"{N}\n")
            arr = generate_data_array(N)
            for x in arr:
                f.write(f"{x}\n")

    elif cfg["TYPE"] == "DEPTHWISE_SEPARABLE":
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

    elif (
        cfg["TYPE"] == "IDENTITY" or cfg["TYPE"] == "POOLING" or cfg["TYPE"] == "CONCAT"
    ):
        # No need to generate for identity layer.
        return
    else:
        raise NotImplementedError(f'Type {cfg["TYPE"]} not recognized.')


def generate_data(cfg: Dict, root_dir: str, key: str) -> str:
    """Generate coefficient data."""

    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    data_file = os.path.join(data_dir, f"data-{key}.txt")
    if os.path.isfile(data_file):
        # return data_file
        os.remove(data_file)

    logger.info(f"Data generated to file: {data_file}")
    if "layers" in cfg:
        for key, value in cfg["layers"].items():
            generate_data_layer(name=key, cfg=value, data_file=data_file)
    elif "NUM_LAYER" not in cfg:
        generate_data_layer(cfg["NAME"], cfg, data_file)
    else:
        for i in range(cfg["NUM_LAYER"]):
            generate_data_layer(cfg["NAME"] + str(i), cfg, data_file)

    return data_file


def get_root_dir(cfg_file: str, cfg: Dict, parent_dir: str = ""):
    if not parent_dir:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "global" in cfg:
        return os.path.join(parent_dir, "build", cfg["NAME"])
    return parent_dir


class CommandRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logger

        self.cfg = toml.load(args.cfg)
        self.logger.info(f"Loaded config: {self.cfg}")

        self.root_dir = get_root_dir(args.cfg, self.cfg, parent_dir=args.root_dir)
        self.log_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.extra_flags = {}
        if hasattr(self.args, "freq") and self.args.freq > 0:
            self.extra_flags["FREQ"] = self.args.freq

    @property
    def cmd(self):
        return ""

    @property
    def key(self):
        key = os.path.basename(self.args.cfg).split(".")[0]
        for k, v in self.extra_flags.items():
            key += f"_{k}-{v}"
        return key

    def get_log_file(self, type: str) -> str:
        return os.path.join(self.log_dir, f"{self.key}.{self.cmd}.{type}.log")

    def get_cfg_for_make(self):
        # overwrite parameters
        cfg_ = copy.deepcopy(self.cfg)
        if "global" in cfg_:
            cfg_ = cfg_["global"]
        if hasattr(self.args, "dbg") and self.args.dbg:
            cfg_["DEBUG"] = "true"
        if hasattr(self.args, "freq") and self.args.freq > 0:
            cfg_["FREQ"] = self.args.freq
        if hasattr(self.args, "sim_device_id"):
            cfg_["SIMDEVICEID"] = self.args.sim_device_id

        return cfg_

    def get_cmd(self, run_golden: bool = False, **kwargs) -> List[str]:
        # Generate data file.
        data_file = generate_data(self.cfg, self.root_dir, key=self.key)
        cli_options = f"-n 2 -f {data_file} {'-g' if run_golden else ''}"

        cfg_for_make = self.get_cfg_for_make()
        cmd = ["make", self.cmd]
        for k, v in cfg_for_make.items():
            cmd.append(f"{k}={v}")
        cmd.append(f"COEFF_FILE={data_file}")
        cmd.append(f"CLI_OPTIONS={cli_options}")
        for k, v in kwargs.items():
            cmd.append(f"{k}={v}")

        return cmd

    def run(self):
        cmd = self.get_cmd()
        logger.info(f"Command to run: {cmd}")
        ret = subprocess.run(
            cmd,
            cwd=self.root_dir,
            stdout=open(self.get_log_file("stdout"), "w"),
            stderr=open(self.get_log_file("stderr"), "w"),
        )
        if ret.returncode != 0:
            logger.error(f"Failed command: {cmd}, return code: {ret.returncode}")
        else:
            logger.info(f"Succeeded command: {cmd}, return code: {ret.returncode}")


class RunsimCommandRunner(CommandRunner):
    @property
    def cmd(self):
        return "runsim"

    @property
    def key(self):
        return os.path.basename(self.args.cfg).split(".")[0]

    def get_cmd(self, **kwargs) -> List[str]:
        cmd = super().get_cmd(run_golden=True, **kwargs)
        return cmd


def runsim(args: argparse.Namespace):
    """Run simulation."""
    RunsimCommandRunner(args).run()


class BuildCommandRunner(CommandRunner):
    @property
    def cmd(self):
        return "build"


def build(args: argparse.Namespace):
    BuildCommandRunner(args).run()


class RunCommandRunner(CommandRunner):
    @property
    def cmd(self):
        return "run"

    def get_cmd(self, **kwargs) -> List[str]:
        cmd = super().get_cmd(**kwargs)
        return cmd


def run(args):
    RunCommandRunner(args).run()


# -------------------------------------------------------------------------


class AppGenerator:
    def __init__(self, args):
        self.args = args
        self.logger = logger

        self.cfg = toml.load(args.cfg)
        self.logger.info(self.cfg)

        if not args.root_dir:
            self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.root_dir = args.root_dir
        self.logger.info(f"Root directory: {self.root_dir}")

        self.name = self.cfg["NAME"]
        assert self.name

        self.src_dir = os.path.join(self.root_dir, "src", self.name)
        os.makedirs(self.src_dir, exist_ok=True)
        self.logger.info(f"Generated src directory: {self.src_dir}")

        self.build_dir = os.path.join(self.root_dir, "build", self.name)
        os.makedirs(self.build_dir, exist_ok=True)
        self.logger.info(f"Generated build directory: {self.build_dir}")

    def read_template(self, file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines)

    def write_template(self, content: str, file_path: str):
        with open(file_path, "w") as f:
            f.write(content)

    @property
    def PRJ(self):
        """From snake to camel case"""
        return "".join(word.title() for word in self.name.split("_"))

    def generate_from_template(self, template_file: str, target_file: str, **kwargs):
        self.write_template(
            self.read_template(template_file).format(**kwargs), target_file
        )

    def generate_makefile(self):
        """Generate Makefile from template."""
        target_file = os.path.join(self.build_dir, "Makefile")
        source_file = os.path.join(TMPL_PARENT_DIR, "build", "TEMPLATE", "Makefile")
        self.generate_from_template(
            source_file,
            target_file,
            PRJ=self.PRJ,
            APPPKG=self.name,
            EXTRA_CFLAGS="",
            WHEREISROOT=os.path.abspath(os.path.dirname(TMPL_PARENT_DIR)),
            DEFS="",
            BUILD_PARAMS="",
            BUILD_NAME_OPTION="",
        )

        self.logger.info(f"Write to makefile: {target_file}")

    def generate_engine_parameters(self):
        source_file = os.path.join(
            TMPL_PARENT_DIR, "src", "TEMPLATE", "ModelEngineParameters.java.tmpl"
        )
        target_file = os.path.join(self.src_dir, f"{self.PRJ}EngineParameters.java")
        self.generate_from_template(
            source_file, target_file, APPPKG=self.name, PRJ=self.PRJ
        )

        self.logger.info(f"Write to engine parameters: {target_file}")

    def generate_manager(self):
        def get_parallel_cfg(key: str, cfg: Dict) -> str:
            ret = ""
            # port_key = "INPUT" if key == "P_C" else "OUTPUT"
            if key not in cfg:
                cfg[key] = [1]
            if not isinstance(cfg[key], list):
                cfg[key] = [cfg[key]]
            name = "".join(key.split("_"))
            for pf in cfg[key]:
                ret += f".{name}({pf})"
            return ret

        cps = ""
        for key, cfg in self.cfg["layers"].items():
            input_cfg = ""
            if "INPUT" in cfg:
                if not isinstance(cfg["INPUT"], list):
                    cfg["INPUT"] = [cfg["INPUT"]]
                for input_name in cfg["INPUT"]:
                    input_cfg += f""".input("{input_name}")"""
            if not input_cfg:
                input_cfg += """.input("")"""

            output_cfg = ""
            if "OUTPUT" in cfg:
                if not isinstance(cfg["OUTPUT"], list):
                    cfg["OUTPUT"] = [cfg["OUTPUT"]]
                for output in cfg["OUTPUT"]:
                    output_type = output.split("_")[0].upper()
                    output_index = int(output.split("_")[1]) if "_" in output else 0
                    output_cfg += f""".output(new Output(OutputType.{output_type}, {output_index}))"""
            cps += f"""
    cps.add(new ConvLayerParameters
                .Builder({cfg['H']}, {cfg['W']}, {cfg['C']}, {cfg['F']}, {cfg['K']}){input_cfg}{output_cfg}
                .BW({self.cfg['global']['BW']})
                .WBW({self.cfg['global']['WBW']})
                .numFracBits({self.cfg['global']['NUM_FRAC_BITS']})
                .type(Type.{cfg['TYPE']})
                .name("{key}")
                .pad({cfg['P']})
                .stride({cfg['S']})
                .seq(CompSeq.values()[{cfg['SEQ'] if 'SEQ' in cfg else 0}])
                .dbg(params.getDebug())
                .coeffOnChip({str(self.cfg['global']['COEFF_ON_CHIP']).lower()})
                .coeffFile(params.getCoeffFile())
                .residual("{cfg['RESIDUAL'] if 'RESIDUAL' in cfg else ""}"){get_parallel_cfg("P_F", cfg)}{get_parallel_cfg("P_C", cfg)}
                .PK({cfg['P_K'] if 'P_K' in cfg else 1})
                .namedRegion("{cfg['NAMED_REGION'] if 'NAMED_REGION' in cfg else ""}")
                .pooling(Pooling.{cfg['POOLING'] if 'POOLING' in cfg else 'MAX'})
                .build());
            """

        source_file = os.path.join(
            TMPL_PARENT_DIR, "src", "TEMPLATE", "ModelManager.java.tmpl"
        )
        target_file = os.path.join(self.src_dir, f"{self.PRJ}Manager.java")
        self.generate_from_template(
            source_file,
            target_file,
            APPPKG=self.name,
            PRJ=self.PRJ,
            FREQ=self.cfg["global"]["FREQ"],
            USE_DRAM=str(self.cfg["global"]["USE_DRAM"]).lower(),
            CPS=cps,
        )
        os.system(f"clang-format -i {target_file}")

        self.logger.info(f"Write to manager: {target_file}")

    def generate_cpucode(self):
        cps_init = ""
        for k in self.cfg["layers"]:
            cps_init += f"""\tcps.push_back(ConvLayerParameters(max_file, "{k}"));\n"""

        source_file = os.path.join(
            TMPL_PARENT_DIR, "src", "TEMPLATE", "ModelCpuCode.cpp.tmpl"
        )
        target_file = os.path.join(self.src_dir, f"{self.PRJ}CpuCode.cpp")
        self.generate_from_template(
            source_file,
            target_file,
            PRJ=self.PRJ,
            CPS_INIT=cps_init,
            DATA_TYPE=f"int{self.cfg['global']['BW']}_t",
        )

        os.system(f"clang-format -i {target_file}")
        self.logger.info(f"Write to CPU code: {target_file}")

    def run(self):
        self.generate_makefile()
        if not self.args.cpu_only:
            self.generate_engine_parameters()
            self.generate_manager()
        self.generate_cpucode()


def gen(args):
    """Generate an application."""
    logger.info(args.cfg)
    AppGenerator(args).run()


# -------------------------------------------------------------------------


def process(args):
    args.cfg = args.cfg[args.index]  # overwrite the list
    args.func(args)


def main():
    parser = argparse.ArgumentParser(description="Deacon tool.")
    parser.add_argument("--cfg", type=str, help="Comma-separated configuration files")
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of parallelised jobs"
    )
    parser.add_argument("-i", "--index", type=int, default=0, help="Job index")
    parser.add_argument("-d", "--root-dir", type=str, default="", help="Root directory")

    subparsers = parser.add_subparsers()

    runsim_parser = subparsers.add_parser("runsim", help="Run simulation")
    runsim_parser.add_argument("--dbg", action="store_true", help="Debug flag")
    runsim_parser.add_argument(
        "--sim-device-id", type=str, default="a", help="Simulation device ID"
    )
    runsim_parser.set_defaults(func=runsim)

    build_parser = subparsers.add_parser("build", help="Run build")
    build_parser.add_argument(
        "--freq", type=int, default=0, help="Overwrite clock frequency"
    )
    build_parser.set_defaults(func=build)

    run_parser = subparsers.add_parser("run", help="Run build")
    run_parser.add_argument(
        "--freq", type=int, default=0, help="Overwrite clock frequency"
    )
    run_parser.set_defaults(func=run)

    gen_parser = subparsers.add_parser(
        "gen", help="Generate an application from a network config."
    )
    gen_parser.add_argument(
        "--cpu-only", action="store_true", help="Only generate the CPU code."
    )
    gen_parser.set_defaults(func=gen)

    args = parser.parse_args()

    if os.path.isdir(args.cfg):
        args.cfg = ",".join([os.path.join(args.cfg, f) for f in os.listdir(args.cfg)])

    args.cfg = args.cfg.split(",")
    args.root_dir = os.path.abspath(args.root_dir)

    # Prepare parallel runs
    sim_device_id = args.sim_device_id if hasattr(args, "sim_device_id") else "a"
    args_list = [copy.deepcopy(args) for i in range(len(args.cfg))]
    for i, args in enumerate(args_list):
        args.index = i
        args.sim_device_id = chr(ord(sim_device_id) + i)

    with mp.Pool(args.jobs) as p:
        p.map(process, args_list)


if __name__ == "__main__":
    main()
