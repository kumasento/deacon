import functools
import multiprocessing as mp
import os
import subprocess
from typing import List, Tuple

DIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(DIR, "logs")


def runsim(workitem: Tuple[List[int], str]):
    cfg = workitem[0]
    id_ = workitem[1]

    cmd = f"make runsim SIMDEVICEID={id_} PAD={cfg[0]}"
    print(f"==> Running: {cmd}")
    stdout_log_file = os.path.join(LOGDIR, f"runsim.pad-{cfg[0]}.stdout.log")
    stderr_log_file = os.path.join(LOGDIR, f"runsim.pad-{cfg[0]}.stderr.log")
    ret = subprocess.run(
        cmd.split(),
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )
    print(f"==> {cmd} DONE - ret: {ret.returncode}")


def compile(cfg: List[int]):
    cmd = f"make build PAD={cfg[0]}"
    print(f"==> Running: {cmd}")
    stdout_log_file = os.path.join(LOGDIR, f"build.pad-{cfg[0]}.stdout.log")
    stderr_log_file = os.path.join(LOGDIR, f"build.pad-{cfg[0]}.stderr.log")
    ret = subprocess.run(
        cmd.split(),
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )
    print(f"==> {cmd} DONE - ret: {ret.returncode}")


def main():
    if not os.path.isdir(LOGDIR):
        os.makedirs(LOGDIR)

    worklist = [[0], [1]]
    ids = ["a", "b"]

    with mp.Pool(len(worklist)) as p:
        p.map(runsim, zip(worklist, ids))

    with mp.Pool(len(worklist)) as p:
        p.map(compile, worklist)


if __name__ == "__main__":
    main()
