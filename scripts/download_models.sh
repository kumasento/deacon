#!/usr/bin/env bash

wget -O evaluation/model_zoo/mobilenet_v1.onnx https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx?download=1
wget -O evaluation/model_zoo/squeezenet1_1.onnx https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
wget -O evaluation/model_zoo/mobilenet_v2.onnx https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
wget -O evaluation/model_zoo/resnet_18.onnx https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx
