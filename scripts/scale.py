"""Scale up a converted graph."""

import argparse
import os
import pprint
from typing import Pattern

from numpy.core.fromnumeric import shape
from pydeacon.graph import DeaconGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--cfg-file", type=str, help="Graph file.")
    parser.add_argument("-o", "--out-file", type=str, help="Output file.")
    parser.add_argument("-s", "--scale", type=int, help="Scale factor.")
    args = parser.parse_args()

    G = DeaconGraph()
    G.load(args.cfg_file)

    pprint.pprint(G.node_map)

    input_node = G.input_node
    assert input_node

    Q = [input_node]
    vis = set()
    vis.add(input_node.name)
    while Q:
        node = Q.pop(0)
        print(node.name)

        for i, input_name in enumerate(node.inputs):
            output_node, j = G.get_output(input_name)
            node.par.P_C[i] = output_node.par.P_F[j]

        for i, output_name in enumerate(node.output_nodes):
            node.par.P_F[i] *= args.scale

            if output_name is None:
                continue
            if output_name in vis:
                continue
            vis.add(output_name)
            Q.append(G.node_map[output_name])

    G.name += f"_s{args.scale}"
    G.sanity_check()
    G.dump(args.out_file)


if __name__ == "__main__":
    main()
