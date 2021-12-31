"""Import manually tuned configuration."""
import argparse
import os

import pandas as pd
from pydeacon.graph import DeaconGraph, LayerType, OutputType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--import-file", type=str)
    parser.add_argument("-f", "--cfg-file", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.import_file, delimiter="\t")
    df = df[["name", "P_C", "P_F"]].set_index("name")
    print(df)
    manual_cfg = df.to_dict("index")
    print(manual_cfg)

    G = DeaconGraph()
    G.load(args.cfg_file)

    for node in G.nodes:
        assert node.name in manual_cfg
        node.par.P_C[0] = int(manual_cfg[node.name]["P_C"])
        node.par.P_F[0] = int(manual_cfg[node.name]["P_F"])

        # NOTE: only workable for mobilenet-v2/squeezenet for now
        if len(node.outputs) == 1:
            if node.layer_type == LayerType.CONCAT:
                node.par.P_C[1] = node.par.P_C[0]
        else:
            if node.outputs[1].output_type == OutputType.IFMAP:
                node.par.P_F[1] = node.par.P_C[0]
            if node.outputs[1].output_type == OutputType.OFMAP:
                node.par.P_F[1] = node.par.P_F[0]

    G.name += "_" + os.path.basename(args.import_file).split(".")[0].split("-")[1]
    G.dump(
        os.path.join(
            os.path.dirname(args.import_file),
            os.path.basename(args.import_file).split(".")[0] + ".toml",
        )
    )


if __name__ == "__main__":
    main()
