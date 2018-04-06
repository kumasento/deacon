import numpy as np
import math
from layer import BaseLayer, ConvLayer, MatmulLayer, STD, PNT, DPT
from net import ConvNet

BIT_WIDTH = 16


def int_ceil(a, b):
  return int(math.ceil(float(a) / b))


class StratixVPlatform(object):
  DSP_BIT_WIDTH = 18
  BRAM_BLOCK_SIZE = 20 * 1024

  @staticmethod
  def get_DSP(num_mult, bit_width):
    return int_ceil(num_mult * bit_width, StratixVPlatform.DSP_BIT_WIDTH)

  @staticmethod
  def get_BRAM(size):
    return int_ceil(size, StratixVPlatform.BRAM_BLOCK_SIZE)


class DesignResourceUsage(object):
  def __init__(self, LUT=0, FF=0, BRAM=0, DSP=0):
    self.LUT = int(LUT)
    self.FF = int(FF)
    self.BRAM = int(BRAM)
    self.DSP = int(DSP)

  def __str__(self):
    return '<%6d, %6d, %6d, %6d>' % (self.LUT, self.FF, self.BRAM, self.DSP)

  def __repr__(self):
    return self.__str__()


class DesignParams(object):
  def __init__(self, P_H=1, P_W=1, P_C=1, P_F=1, T_H=1, T_W=1, T_C=1, T_F=1,
               P_ROW=1, P_COL=1, T_ROW=1, T_COL=1, K=3):
    self.P_H = P_H
    self.P_W = P_W
    self.P_C = P_C
    self.P_F = P_F
    self.P_ROW = P_ROW
    self.P_COL = P_COL

    self.T_H = T_H
    self.T_W = T_W
    self.T_C = T_C
    self.T_F = T_F
    self.T_ROW = T_ROW
    self.T_COL = T_COL

    self.K = K


class Design(object):

  def __init__(self, params: DesignParams):
    self.params = params

  def run(self, net: ConvNet):
    pass


class BasicDesign(Design):

  def run_std_conv_tile(self):
    num_cycle = [int_ceil(self.params.T_H, self.params.P_H), int_ceil(self.params.T_W, self.params.P_W),
                 int_ceil(self.params.T_C, self.params.P_C), int_ceil(self.params.T_F, self.params.P_F)]
    total_num_cycles = np.prod(num_cycle)

    return total_num_cycles

  def run_std_conv(self, layer: ConvLayer):
    num_tile = [int_ceil(layer.H, self.params.T_H), int_ceil(layer.W, self.params.T_W),
                int_ceil(layer.C, self.params.T_C), int_ceil(layer.F, self.params.T_F)]
    total_num_tiles = np.prod(num_tile)

    return self.run_std_conv_tile() * total_num_tiles

  def run_conv(self, layer: ConvLayer):
    if layer.T == STD:
      return self.run_std_conv(layer)

  def run_matmul_tile(self):
    num_cycle = [int_ceil(self.params.T_ROW, self.params.P_ROW),
                 int_ceil(self.params.T_COL, self.params.P_COL)]
    total_num_cycles = np.prod(num_cycle)

    return total_num_cycles

  def run_matmul(self, layer: MatmulLayer):
    num_tile = [int_ceil(layer.num_cols, self.params.T_COL),
                int_ceil(layer.num_rows, self.params.T_ROW)]
    total_num_tiles = np.prod(num_tile)

    return total_num_tiles * self.run_matmul_tile()

  def run_layer(self, layer: BaseLayer):
    if isinstance(layer, ConvLayer):
      return self.run_conv(layer)
    if isinstance(layer, MatmulLayer):
      return self.run_matmul(layer)

    raise RuntimeError("Type of layer %s is not supported: %s",
                       layer.name, type(layer))

  def run(self, net: ConvNet):
    return np.sum([self.run_layer(layer) for layer in net.layers])

  @property
  def resource(self):
    mem_elems = np.array([(self.params.T_C * self.params.T_H * self.params.T_W),
                          (self.params.T_W * self.params.K * self.params.P_C),
                          (self.params.P_F * self.params.T_H * self.params.T_W)])
    mem_size = mem_elems * BIT_WIDTH
    num_mult = (self.params.P_C * self.params.P_W *
                self.params.P_F * (self.params.K ** 2))

    return DesignResourceUsage(BRAM=np.sum([StratixVPlatform.get_BRAM(s) for s in mem_size]),
                               DSP=StratixVPlatform.get_DSP(num_mult, BIT_WIDTH))
