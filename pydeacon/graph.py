"""Graph data structure used in Deacon."""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import toml


class LayerType(Enum):
    STANDARD = 0
    DEPTHWISE_SEPARABLE = 1
    POOLING = 2
    CONCAT = 3
    DEPTHWISE = 4


@dataclass
class Shape:
    H: int  # output height
    W: int  # output width
    C: int  # input channels
    F: int  # output channels


class OutputType(Enum):
    OFMAP = 0
    IFMAP = 1


@dataclass
class Output:
    output_type: OutputType
    index: int = 0

    def __post_init__(self):
        if isinstance(self.index, str):
            try:
                self.index = int(self.index)
            except ValueError:
                self.index = 0


class Seq(Enum):
    FILTER_MAJOR = 0
    CHANNEL_MAJOR = 1


@dataclass
class Parallelism:
    P_C: List[int] = field(default_factory=list)
    P_F: List[int] = field(default_factory=list)


@dataclass
class Node:
    shape: Shape
    name: str
    layer_type: str
    K: int  # kernel shape
    S: int  # stride
    P: int  # padding
    seq: Seq  # compute sequence
    par: Parallelism = None  # parallelism
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    residual: Optional[str] = None


@dataclass
class Globals:
    a_bw: int  # activation bit width
    w_bw: int  # weight bit width
    num_frac_bits: int  # number of fraction bits
    freq: int  # default frequency
    coeff_on_chip: bool  # store coefficient on-chip
    use_dram: bool  # whether to use DRAM


@dataclass
class DeaconGraph:
    name: str = ""
    globals: Globals = None
    node_map: Dict[str, Node] = field(default_factory=dict)

    @staticmethod
    def get_inputs(layer_cfg: Dict, prev_node: Optional[Node]) -> List[str]:
        inputs = layer_cfg["INPUT"] if "INPUT" in layer_cfg else []
        if not inputs and prev_node:
            inputs = [prev_node.name]
        return inputs

    @staticmethod
    def get_outputs(layer_cfg: Dict) -> List[Output]:
        outputs = layer_cfg["OUTPUT"] if "OUTPUT" in layer_cfg else []
        if not outputs:
            outputs = ["OFMAP"]
        return [
            Output(output_type=OutputType[x.split("_")[0]], index=x.split("_")[-1])
            for x in outputs
        ]

    def get_output(self, key: str) -> Tuple[Node, int]:
        splits = key.split("_")
        return self.node_map[splits[0]], int(splits[1]) if len(splits) == 2 else 0

    def initialize_parallelism(self, node: Node):
        """Set the minimum required parallelism to balance the stream rate."""
        if node.par:
            return
        P_C = []
        for input_name in node.inputs:
            in_node, in_port = self.get_output(input_name)
            P_C.append(in_node.par.P_F[in_port])
        if not P_C:
            P_C = [1]

        P_F = [1] * len(node.outputs)

        if node.layer_type == LayerType.CONCAT:
            assert len(np.unique(P_C)) == 1
            P_F = [P_C[0] * len(node.inputs)] * len(node.outputs)
        elif node.layer_type == LayerType.POOLING:
            P_F = list(P_C)

        node.par = Parallelism(P_C=P_C, P_F=P_F)

    def sanity_check(self):
        # check shape
        for node in self.node_map.values():
            if (
                node.layer_type == LayerType.STANDARD
                or node.layer_type == LayerType.DEPTHWISE_SEPARABLE
                or node.layer_type == LayerType.POOLING
            ):
                if node.P == 1 and node.K == 3 and node.S == 2:
                    continue  # has been handled specifically in hardware
                if node.P == 3 and node.K == 7 and node.S == 2:
                    continue  # has been handled specifically in hardware
                assert (node.shape.H + 2 * node.P - node.K) % node.S == 0
                assert (node.shape.W + 2 * node.P - node.K) % node.S == 0

        # check parallelism
        for node in self.nodes:
            if node.layer_type == LayerType.CONCAT:
                assert len(np.unique(node.par.P_C)) == 1
                assert all([pf == sum(node.par.P_C) for pf in node.par.P_F])

            for i, input_name in enumerate(node.inputs):
                output_node, j = self.get_output(input_name)
                assert node.par.P_C[i] == output_node.par.P_F[j]

            if node.residual:
                output_node, j = self.get_output(node.residual)
                # print(node.par.P_F[0], output_node.par.P_F[j])
                assert node.par.P_F[0] == output_node.par.P_F[j]

    def initialize_seq(self):
        """Based on a simplest greedy algorithm. Still improving"""
        input_node = None
        for node in self.node_map.values():
            if not node.inputs:
                input_node = node
                break

        input_node.seq = Seq.FILTER_MAJOR
        vis = set()
        Q = [input_node]
        while Q:
            node = Q.pop(0)
            vis.add(node.name)

            for output_node in node.output_nodes:
                if output_node in vis:
                    assert self.node_map[output_node].seq != node.seq
                    continue

                self.node_map[output_node].seq = (
                    Seq.FILTER_MAJOR
                    if node.seq == Seq.CHANNEL_MAJOR
                    else Seq.CHANNEL_MAJOR
                )
                Q.append(self.node_map[output_node])

    @property
    def input_node(self) -> Optional[Node]:
        for node in self.nodes:
            if not node.inputs:
                return node
        return None

    @property
    def nodes(self) -> List[Node]:
        vis = set()
        nodes = []
        for node in self.node_map.values():
            if node.name in vis:
                continue
            nodes.append(node)
        return nodes

    def load(self, cfg_file: str):
        cfg = toml.load(cfg_file)
        self.name = cfg["NAME"]

        # load globals
        self.globals = Globals(
            a_bw=cfg["global"]["BW"],
            w_bw=cfg["global"]["WBW"],
            freq=cfg["global"]["FREQ"],
            num_frac_bits=cfg["global"]["NUM_FRAC_BITS"],
            coeff_on_chip=cfg["global"]["COEFF_ON_CHIP"],
            use_dram=cfg["global"]["USE_DRAM"],
        )

        prev_node = None
        for layer_name, layer_cfg in cfg["layers"].items():
            node = Node(
                shape=Shape(
                    H=layer_cfg["H"],
                    W=layer_cfg["W"],
                    C=layer_cfg["C"],
                    F=layer_cfg["F"],
                ),
                name=layer_name,
                layer_type=LayerType[layer_cfg["TYPE"]],
                K=layer_cfg["K"],
                S=layer_cfg["S"],
                P=layer_cfg["P"],
                inputs=self.get_inputs(layer_cfg, prev_node=prev_node),
                outputs=self.get_outputs(layer_cfg),
                residual=layer_cfg["RESIDUAL"] if "RESIDUAL" in layer_cfg else None,
                seq=Seq(0 if "SEQ" not in layer_cfg else layer_cfg["SEQ"]),
            )
            node.output_nodes = [None] * len(node.outputs)
            self.initialize_parallelism(node)

            prev_node = node
            self.node_map[layer_name] = node

        for node in self.nodes:
            for input_name in node.inputs:
                output_node, index = self.get_output(input_name)
                output_node.output_nodes[index] = node.name

    def dump(self, cfg_file: str):
        cfg = {
            "NAME": self.name,
            "global": {
                "BW": self.globals.a_bw,
                "WBW": self.globals.w_bw,
                "FREQ": self.globals.freq,
                "NUM_FRAC_BITS": self.globals.num_frac_bits,
                "COEFF_ON_CHIP": self.globals.coeff_on_chip,
                "USE_DRAM": self.globals.use_dram,
            },
        }
        cfg["layers"] = {}
        for node in self.node_map.values():
            cfg["layers"][node.name] = {
                "H": node.shape.H,
                "W": node.shape.W,
                "C": node.shape.C,
                "F": node.shape.F,
                "K": node.K,
                "S": node.S,
                "P": node.P,
                "P_C": node.par.P_C,
                "P_F": node.par.P_F,
                "SEQ": node.seq.value,
                "TYPE": node.layer_type.name,
                "INPUT": node.inputs,
                "OUTPUT": [
                    x.output_type.name + ("" if x.index == 0 else f"_{x.index}")
                    for x in node.outputs
                ],
                "RESIDUAL": node.residual,
            }

        self.sanity_check()
        with open(cfg_file, "w") as f:
            toml.dump(cfg, f)

