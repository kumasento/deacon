import numpy as np

from layer import ConvLayer, STD, DPT, PNT, MatmulLayer


class ConvNet:

  def __init__(self, layers):
    self.layers = layers

  @property
  def num_params(self):
    return np.sum([layer.num_params for layer in self.layers])

  @property
  def num_ops(self):
    return np.sum([layer.num_ops for layer in self.layers])

  def __str__(self):
    return '\n'.join(['%s' % layer for layer in self.layers])

  def __repr__(self):
    return self.__str__()


VGG16 = ConvNet([
    ConvLayer("conv1_1", 224, 224, 3, 64, 3, STD),
    ConvLayer("conv1_2", 224, 224, 64, 64, 3, STD),
    ConvLayer("conv2_1", 112, 112, 64, 128, 3, STD),
    ConvLayer("conv2_2", 112, 112, 128, 128, 3, STD),
    ConvLayer("conv3_1", 56, 56, 128, 256, 3, STD),
    ConvLayer("conv3_2", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv3_3", 56, 56, 256, 256, 3, STD),
    ConvLayer("conv4_1", 28, 28, 256, 512, 3, STD),
    ConvLayer("conv4_2", 28, 28, 512, 512, 3, STD),
    ConvLayer("conv4_3", 28, 28, 512, 512, 3, STD),
    ConvLayer("conv5_1", 14, 14, 512, 512, 3, STD),
    ConvLayer("conv5_2", 14, 14, 512, 512, 3, STD),
    ConvLayer("conv5_3", 14, 14, 512, 512, 3, STD),
    MatmulLayer("fc6", 7 * 7 * 512, 4096),
    MatmulLayer("fc7", 4096, 4096),
    MatmulLayer("fc8", 4096, 1000),
])
