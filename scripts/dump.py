"""Dump spreadsheet"""
import argparse
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from posixpath import basename
from typing import Dict

import numpy as np
import onnx
import toml
from onnx import shape_inference
from pydeacon.graph import (
    DeaconGraph,
    Globals,
    LayerType,
    Node,
    Output,
    OutputType,
    Parallelism,
    Seq,
    Shape,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model-file", type=str, help="Model file")
    args = parser.parse_args()

    G = DeaconGraph()
    G.load(args.model_file)

    G.dump_spreadsheet(
        os.path.join(
            os.path.dirname(args.model_file),
            os.path.basename(args.model_file).split(".")[0] + ".csv",
        )
    )


if __name__ == "__main__":
    main()
