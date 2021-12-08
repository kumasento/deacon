import multiprocessing as mp
import os
import subprocess
from typing import List


def compile(seqs: List[int]):
    cmd = f"make build COEFF_ON_CHIP=true USE_DRAM=true SEQ0={seqs[0]} SEQ1={seqs[1]}"
    print(f"==> Running: {cmd}")
    stdout_log_file = f"build.seq-{seqs[0]}-{seqs[1]}.stdout.log"
    stderr_log_file = f"build.seq-{seqs[0]}-{seqs[1]}.stderr.log"
    subprocess.run(
        [
            "make",
            "build",
            "COEFF_ON_CHIP=true",
            "USE_DRAM=true",
            f"SEQ0={seqs[0]}",
            f"SEQ1={seqs[1]}",
        ],
        stdout=open(stdout_log_file, "w"),
        stderr=open(stderr_log_file, "w"),
    )


def main():
    with mp.Pool(4) as p:
        p.map(compile, [[0, 0], [0, 1], [1, 0], [1, 1]])


if __name__ == "__main__":
    main()
