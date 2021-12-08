import os
import re
import subprocess
from collections import defaultdict
from typing import List

import pandas as pd

BASE_DIR = "/mnt/ccnas2/bdp/rz3515/maxcompiler_builds"
DESIGN_NAME = "ConvTwoLayers_MAIA_DFE_b16_H32_W32_C32_F32_K3_f1_c1_k1_SEQ{seq0}_{seq1}_DRAM_COC_FREQ_200"
CONFIGS = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]


def parse_build_log(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    data = {}

    i = next(i for i, l in enumerate(lines) if "FINAL RESOURCE" in l)
    for line in lines[i + 1 : i + 8]:
        line = line[line.index("PROGRESS:") + len("PROGRESS:") :]
        line = line.strip()
        m = re.search("\s+\d+ / \d+", line)
        if m:
            resource = line.split(":")[0].strip()
            usage = line.split(":")[1].strip()
            used = usage.split("/")[0].strip()
            # total = usage.split("/")[1].split("(")[0].strip()
            data[resource] = used

    return data


def get_perf(cfg: List[int]):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    print(
        subprocess.run(
            [
                "ssh",
                "-t",
                "lima01",
                f'cd {dir_path}; zsh -ic "make run COEFF_ON_CHIP=true USE_DRAM=true SEQ0={cfg[0]} SEQ1={cfg[1]}"',
            ],
            capture_output=True,
        )
    )


def main():
    data = defaultdict(list)
    for cfg in CONFIGS:
        data["seq0"].append(cfg[0])
        data["seq1"].append(cfg[1])

        path = os.path.join(
            BASE_DIR, DESIGN_NAME.format(seq0=cfg[0], seq1=cfg[1]), "_build.log"
        )
        res = parse_build_log(path)

        for key, value in res.items():
            data[key].append(value)

        get_perf(cfg)

    df = pd.DataFrame(data)
    print(df)

    df.to_csv("resource-usage.csv")


if __name__ == "__main__":
    main()
