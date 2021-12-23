#!/usr/bin/env python3

""" Convert from pre-trained models. """

import argparse
import os
from collections import OrderedDict

import numpy as np
import onnx
import toml
from onnx import shape_inference


class ONNXConverter:
    def __init__(self):
        pass

    def convert_name(self, name: str):
        return name.replace("_", "").lower()

    def get_shape(self, name: str, value_info: dict):
        shape = value_info[name].type.tensor_type.shape
        return [x.dim_value for x in shape.dim[1:]]

    def convert(self, src_file: str, dst_file: str):
        model = onnx.load(src_file)
        model = shape_inference.infer_shapes(model)

        value_info = {info.name: info for info in model.graph.value_info}
        for input_tensor in model.graph.input:
            value_info[input_tensor.name] = input_tensor

        data = {}
        data["NAME"] = self.convert_name(os.path.basename(src_file).split(".")[0])
        data["layers"] = OrderedDict()
        for node in model.graph.node:
            layer = {}
            if node.op_type == "Conv":
                assert len(node.output) == 1

                out_shape = self.get_shape(node.output[0], value_info)
                in_shape = self.get_shape(node.input[0], value_info)
                layer["H"] = out_shape[1]
                layer["W"] = out_shape[2]
                layer["F"] = out_shape[0]
                layer["C"] = in_shape[0]
                for attr in node.attribute:
                    if attr.name == "kernel_shape":
                        assert np.unique(attr.ints).size == 1
                        layer["K"] = attr.ints[0]
                    if attr.name == "pads":
                        assert np.unique(attr.ints).size == 1
                        layer["P"] = attr.ints[0]
                    if attr.name == "strides":
                        assert np.unique(attr.ints).size == 1
                        layer["S"] = attr.ints[0]

                print(node.input)

            if layer:
                data["layers"][self.convert_name(node.name)] = layer

        with open(dst_file, "w") as f:
            toml.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model-file", type=str, help="Model file")
    parser.add_argument(
        "-o", "--config-file", type=str, help="Dump converted config file"
    )
    args = parser.parse_args()

    ONNXConverter().convert(args.model_file, args.config_file)


if __name__ == "__main__":
    main()
