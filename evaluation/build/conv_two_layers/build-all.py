import multiprocessing as mp
import os
from typing import List


def compile(seqs: List[int]):
    cmd = f"make build COEFF_ON_CHIP=true USE_DRAM=true SEQ0={seqs[0]} SEQ1={seqs[1]}"
    print(f"==> Running: {cmd}")
    os.system(cmd)


def main():
    with mp.Pool(4) as p:
        p.map(compile, [[0, 0], [0, 1], [1, 0], [1, 1]])


if __name__ == "__main__":
    main()
