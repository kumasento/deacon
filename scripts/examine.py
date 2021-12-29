"""Examine the compute sequence of a model."""
import argparse
import os
import pprint

from pydeacon.graph import DeaconGraph


def check_comp_seq(G: DeaconGraph):
    for node in G.node_map.values():
        for input_name in node.inputs:
            input_node = G.node_map[input_name]
            if input_node.seq == node.seq:
                print(f"Violated CompSeq rule: {input_node} and {node}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--cfg", type=str, help="Configuration file.")
    parser.add_argument("-o", "--out", type=str, help="Sanitized resul.")
    args = parser.parse_args()

    G = DeaconGraph()
    G.load(args.cfg)

    pprint.pprint(G.node_map)

    check_comp_seq(G)

    G.dump(args.out)


if __name__ == "__main__":
    main()
