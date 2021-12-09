import functools
import multiprocessing as mp
import os
import subprocess
from typing import List, Tuple

DIR = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(DIR, "logs")


def runsim(workitem: Tuple[List[int], str]):
    seqs = workitem[0]
    id_ = workitem[1]

    cmd = f"make runsim SIMDEVICEID={id_} COEFF_ON_CHIP=true USE_DRAM=true SEQ0={seqs[0]} SEQ1={seqs[1]} SEQ2={seqs[2]}"
    print(f"==> Running: {cmd}")
    stdout_log_file = os.path.join(
        LOGDIR, f"runsim.seq-{seqs[0]}-{seqs[1]}-{seqs[2]}.stdout.log"
    )
    stderr_log_file = os.path.join(
        LOGDIR, f"runsim.seq-{seqs[0]}-{seqs[1]}-{seqs[2]}.stderr.log"
    )
    ret = subprocess.run(
        cmd.split(),
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )
    print(f"==> {cmd} DONE - ret: {ret.returncode}")


def compile(seqs: List[int]):
    cmd = f"make build COEFF_ON_CHIP=true USE_DRAM=true SEQ0={seqs[0]} SEQ1={seqs[1]} SEQ2={seqs[2]}"
    print(f"==> Running: {cmd}")
    stdout_log_file = os.path.join(
        LOGDIR, f"build.seq-{seqs[0]}-{seqs[1]}-{seqs[2]}.stdout.log"
    )
    stderr_log_file = os.path.join(
        LOGDIR, f"build.seq-{seqs[0]}-{seqs[1]}-{seqs[2]}.stderr.log"
    )
    ret = subprocess.run(
        [
            "make",
            "build",
            "COEFF_ON_CHIP=true",
            "USE_DRAM=true",
            f"SEQ0={seqs[0]}",
            f"SEQ1={seqs[1]}",
            f"SEQ2={seqs[2]}",
        ],
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )
    print(f"==> {cmd} DONE - ret: {ret.returncode}")


def main():
    if not os.path.isdir(LOGDIR):
        os.makedirs(LOGDIR)

    worklist = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
    ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    with mp.Pool(len(worklist)) as p:
        p.map(runsim, zip(worklist, ids))

    with mp.Pool(len(worklist)) as p:
        p.map(compile, worklist)


if __name__ == "__main__":
    main()
