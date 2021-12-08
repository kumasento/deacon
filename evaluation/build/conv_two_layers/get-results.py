import os
import re
from collections import defaultdict

import pandas as pd

BASE_DIR = "/mnt/ccnas2/bdp/rz3515/maxcompiler_builds"
DESIGNS = [
    "ConvTwoLayers_MAIA_DFE_b16_H32_W32_C32_F32_K3_f1_c1_k1_SEQ0_0_DRAM_COC_FREQ_200",
    "ConvTwoLayers_MAIA_DFE_b16_H32_W32_C32_F32_K3_f1_c1_k1_SEQ0_1_DRAM_COC_FREQ_200",
    "ConvTwoLayers_MAIA_DFE_b16_H32_W32_C32_F32_K3_f1_c1_k1_SEQ1_0_DRAM_COC_FREQ_200",
    "ConvTwoLayers_MAIA_DFE_b16_H32_W32_C32_F32_K3_f1_c1_k1_SEQ1_1_DRAM_COC_FREQ_200",
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


def main():
    data = defaultdict(list)
    for design in DESIGNS:
        data["name"].append(design)
        path = os.path.join(BASE_DIR, design, "_build.log")
        res = parse_build_log(path)

        for key, value in res.items():
            data[key].append(value)

    df = pd.DataFrame(data)
    print(df)

    df.to_csv("resource-usage.csv")


if __name__ == "__main__":
    main()
