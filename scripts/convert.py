#!/usr/bin/env python3

""" Convert from pre-trained models. """

import argparse
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
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
    Seq,
    Shape,
)


def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n > 0


def next_power_of_two(x: int) -> int:
    return pow(2, math.ceil(math.log(x) / math.log(2)))


@dataclass
class Tensor:
    layer_name: str  # from which layer generates this tensor.
    index: int  # which output port


class ONNXConverter:
    def __init__(self, last_padded: bool = False, bit_width: int = 16):
        self.last_padded = last_padded
        self.bit_width = bit_width

    def convert_name(self, name: str):
        return name.replace("_", "").lower()

    def get_shape(self, name: str, value_info: dict):
        shape = value_info[name].type.tensor_type.shape
        return [x.dim_value for x in shape.dim[1:]]

    def parse_attrs(self, node) -> Dict:
        attrs = {}
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                assert np.unique(attr.ints).size == 1
                attrs["K"] = attr.ints[0]
            if attr.name == "pads":
                assert np.unique(attr.ints).size == 1
                attrs["P"] = attr.ints[0]
            if attr.name == "strides":
                assert np.unique(attr.ints).size == 1
                attrs["S"] = attr.ints[0]
            if attr.name == "group":
                attrs["G"] = attr.i
        return attrs

    def get_config_suffix(self) -> str:
        suffix = "_onnx"
        if self.last_padded:
            suffix += "_last_padded"
        if self.bit_width != 16:
            suffix += f"_b{self.bit_width}"

        return suffix

    def convert(self, src_file: str, dst_file: str):
        model = onnx.load(src_file)
        model = shape_inference.infer_shapes(model)

        G = DeaconGraph(
            name="_".join(os.path.basename(src_file).split(".")[:-1]).replace("-", "_")
            + self.get_config_suffix(),
            globals=Globals(
                a_bw=self.bit_width,
                w_bw=self.bit_width,
                freq=200,
                coeff_on_chip=True,
                use_dram=True,
                num_frac_bits=8 if self.bit_width == 16 else 0,
            ),
        )

        value_info = {info.name: info for info in model.graph.value_info}
        for input_tensor in model.graph.input:
            value_info[input_tensor.name] = input_tensor

        prev_d_node = None
        tensor_map: Dict[str, Tensor] = {}

        for node in model.graph.node:
            d_node = None

            if node.op_type not in [
                "Conv",
                "MaxPool",
                "Relu",
                "Concat",
                "Dropout",
                "Clip",
                "Add",
                "BatchNormalization",
            ]:
                print(node.op_type + " not supported. Break.")
                break

            out_shape = self.get_shape(node.output[0], value_info)
            in_shape = self.get_shape(node.input[0], value_info)
            attrs = self.parse_attrs(node)
            print(node.name, in_shape, " -> ", out_shape)

            new_node = True
            if node.op_type == "Conv":
                assert len(node.output) == 1
                d_node = Node(
                    name=self.convert_name(node.name),
                    shape=Shape(
                        H=out_shape[1], W=out_shape[2], F=out_shape[0], C=in_shape[0]
                    ),
                    K=attrs["K"],
                    P=attrs["P"],
                    S=attrs["S"],
                    seq=Seq.FILTER_MAJOR,
                    layer_type=LayerType.STANDARD
                    if "G" not in attrs or attrs["G"] != in_shape[0]
                    else LayerType.DEPTHWISE,
                )

                # Fuse into depthwise separable
                if node.input[0] in tensor_map:
                    input_name = tensor_map[node.input[0]].layer_name
                    if (
                        d_node.K == 1
                        and G.node_map[input_name].layer_type == LayerType.DEPTHWISE
                    ):
                        p_node = G.node_map[self.convert_name(input_name)]
                        p_node.layer_type = LayerType.DEPTHWISE_SEPARABLE
                        p_node.shape.F = d_node.shape.F
                        d_node = p_node
                        new_node = False

            elif node.op_type == "Add":
                na = G.node_map[tensor_map[node.input[0]].layer_name]
                nb = G.node_map[tensor_map[node.input[1]].layer_name]
                nc = G.node_map[nb.inputs[0]]

                # na --> nc --> nb

                if (
                    nb.layer_type == LayerType.DEPTHWISE_SEPARABLE
                    and nc.inputs[0] == na.name
                    and nc.layer_type == LayerType.STANDARD
                ):  # inverted bottleneck
                    nc.outputs.append(Output(output_type=OutputType.IFMAP))
                    nb.residual = nc.name + "_" + str(len(nc.outputs) - 1)
                elif (
                    nb.layer_type == LayerType.STANDARD
                    and nc.inputs[0] == na.name
                    and nc.layer_type == LayerType.STANDARD
                ):  # resnet-18 stack type
                    nc.outputs.append(Output(output_type=OutputType.IFMAP))
                    nb.residual = nc.name + "_" + str(len(nc.outputs) - 1)
                elif (
                    nb.layer_type == LayerType.STANDARD
                    and nc.layer_type == LayerType.STANDARD
                    and nc.inputs[0].split("_")[0] == na.inputs[0].split("_")[0]
                ):  # resnet-18 shortcut
                    # na is the shortcut convolution
                    # should erase na, add a new input to nb, from a duplicated ifmap of nc, and assign residual to that.
                    nc.outputs.append(Output(output_type=OutputType.IFMAP))
                    extra_input = f"{nc.name}_{len(nc.outputs)-1}"
                    nb.inputs.append(extra_input)
                    nb.residual = extra_input

                    # make sure the first output is taken by nc.
                    if "_" in nc.inputs[0]:
                        nc.inputs[0], na.inputs[0] = na.inputs[0], nc.inputs[0]
                        output_node, index = G.get_output(nc.inputs[0])
                        output_node.outputs[0], output_node.outputs[1] = (
                            output_node.outputs[1],
                            output_node.outputs[0],
                        )
                        output_node.output_nodes[0], output_node.output_nodes[1] = (
                            output_node.output_nodes[1],
                            output_node.output_nodes[0],
                        )
                    G.node_map = {k: v for k, v in G.node_map.items() if v != na}
                    for input_name in na.inputs:
                        output_node, index = G.get_output(input_name)
                        assert index == len(output_node.outputs) - 1
                        output_node.outputs.pop(index)
                        output_node.output_nodes.pop(index)
                else:
                    print("na = ", na)
                    print("nc = ", nc)
                    print("nb = ", nb)
                    assert False

                d_node = nb
                new_node = False

            elif node.op_type == "MaxPool":
                d_node = Node(
                    name=self.convert_name(node.name),
                    shape=Shape(
                        H=out_shape[1], W=out_shape[2], F=out_shape[0], C=in_shape[0]
                    ),
                    K=attrs["K"],
                    P=attrs["P"],
                    S=attrs["S"],
                    seq=Seq.FILTER_MAJOR,
                    layer_type=LayerType.POOLING,
                )

            elif node.op_type in ["Relu", "Dropout", "Clip", "BatchNormalization"]:
                assert prev_d_node
                d_node = prev_d_node
                new_node = False

            elif node.op_type == "Concat":
                d_node = Node(
                    name=self.convert_name(node.name),
                    shape=Shape(
                        H=out_shape[1], W=out_shape[2], F=out_shape[0], C=in_shape[0]
                    ),
                    K=1,
                    P=0,
                    S=1,
                    seq=Seq.FILTER_MAJOR,
                    layer_type=LayerType.CONCAT,
                )

            else:
                assert False

            assert len(node.input) >= 1

            # append the current node as one of the output node of its inputs.
            if new_node:
                if node.input[0] == "data" or node.input[0] == "input":
                    d_node.inputs = []  # first layer
                else:
                    for tensor_name in node.input:
                        if tensor_name not in tensor_map:
                            continue  # possibly a weight tensor
                        input_name = tensor_map[tensor_name].layer_name
                        key = self.convert_name(input_name)
                        if key == d_node.name:  # merged
                            continue
                        if key in G.node_map:
                            input_name = G.node_map[key].name
                            if len(G.node_map[key].output_nodes) >= 1:
                                input_name += "_" + str(
                                    len(G.node_map[key].output_nodes)
                                )
                            d_node.inputs.append(input_name)
                for input_name in d_node.inputs:
                    input_node, index = G.get_output(input_name)
                    input_node.output_nodes.append(d_node.name)

                    assert len(input_node.output_nodes) == index + 1
                    # We assume only the original output will be used.
                    input_node.outputs.append(
                        Output(output_type=OutputType.OFMAP, index=0)
                    )

            # create entries in the tensor map
            for i, tensor_name in enumerate(node.output):
                tensor_map[tensor_name] = Tensor(layer_name=d_node.name, index=i)

            G.node_map[self.convert_name(node.name)] = d_node
            prev_d_node = d_node
            if new_node:
                print(d_node)

        prev_d_node.outputs.append(Output(output_type=OutputType.OFMAP))
        if self.last_padded:
            if prev_d_node and not is_power_of_two(prev_d_node.shape.F):
                print(prev_d_node.shape.F)
                prev_d_node.shape.F = next_power_of_two(prev_d_node.shape.F)

        vis = set()
        for d_node in G.node_map.values():
            if d_node.name in vis:
                continue
            vis.add(d_node.name)
            G.initialize_parallelism(d_node)

        G.initialize_seq()
        G.sanity_check()
        G.dump(dst_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model-file", type=str, help="Model file")
    parser.add_argument(
        "-o", "--config-file", type=str, help="Dump converted config file"
    )
    parser.add_argument(
        "--last-padded",
        action="store_true",
        help="Whether to pad the last layer to power of 2",
    )
    parser.add_argument("--bw", type=int, default=8, help="Bit width")
    args = parser.parse_args()

    ONNXConverter(last_padded=args.last_padded, bit_width=args.bw).convert(
        args.model_file, args.config_file
    )


if __name__ == "__main__":
    main()
