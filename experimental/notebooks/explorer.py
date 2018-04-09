""" Design space exploration.
"""
from net import ConvNet
from design import DesignParams, BasicDesign
from fpga_platform import Platform
from perf_predict import BasicPredictor

import numpy as np
import itertools
from collections import OrderedDict


def factor(n):
  result = []
  for i in range(1, n + 1):
    if n % i == 0:
      result.append(i)
  return result


class Explorer(object):
  def explore(self, net: ConvNet, platform: Platform):
    pass


class BasicExplorer(Explorer):

  def candidates(self, use_winograd=False):
    tile_shapes = OrderedDict({
        "T_H": [7, 14, 28, 56, 112, 224],
        "T_W": [7, 14, 28, 56, 112, 224],
        "T_C": [64, 128, 256, 512, 1024, 2048],
        "T_F": [64, 128, 256, 512, 1024, 2048],
    })

    for T_H,  T_W, T_C, T_F in itertools.product(*list(tile_shapes.values())):
      T_ROW = T_F
      T_COL = T_H * T_W * T_C

      params = OrderedDict({
          'P_H': [1],
          'P_W': [1],
          'P_C': factor(T_C),
          'P_F': factor(T_F)
      })

      for P_H, P_W, P_C, P_F in itertools.product(*list(params.values())):
        if use_winograd:
          P_COL = P_C * 4
          P_ROW = P_F * 9
        else:
          P_COL = P_H * P_W * P_C
          P_ROW = P_F * 9

        yield [T_H, T_W, T_C, T_F, P_H, P_W, P_C, P_F, T_ROW, T_COL, P_ROW, P_COL]

  def explore(self, net: ConvNet, platform: Platform, use_winograd=False):
    predictor = BasicPredictor()

    results = []
    for cfg in self.candidates(use_winograd):
      params = DesignParams(T_H=cfg[0], T_W=cfg[1], T_C=cfg[2], T_F=cfg[3], P_H=cfg[4], P_W=cfg[5],
                            P_C=cfg[6], P_F=cfg[7], T_ROW=cfg[8], T_COL=cfg[9], P_ROW=cfg[10], P_COL=cfg[11])

      design = BasicDesign(params)
      res = design.resource
      BRAM, DSP, ALM = res.BRAM, res.DSP, res.ALM

      if BRAM > platform.max_BRAM or DSP > platform.max_DSP or ALM > platform.max_ALM:
        continue

      num_cycle, elapsed, GFLOPS, _, CTC = predictor.predict(net, design)

      results.append([BRAM, DSP, ALM, num_cycle, elapsed, GFLOPS, CTC, params])

    return np.array(results)


class FusedBlockExplorer(Explorer):

  def bottleneck_candidates(self, use_winograd=False):
    tile_shapes = OrderedDict({
        "T_H": [7, 14, 28, 56, 112, 224],
        "T_W": [7, 14, 28, 56, 112, 224],
        'T_C1': [32, 64, 128, 256, 512, 1024, 2048],
        'T_C2': [32, 64, 128, 256, 512, 1024, 2048],
        'T_F': [32, 64, 128, 256, 512, 1024, 2048],
    })

    for T_H, T_W, T_C1, T_C2, T_F in itertools.product(*list(tile_shapes.values())):
      T_ROW = T_F * T_C2
      T_COL = T_H * T_W * T_C1

      params = OrderedDict({
          'P_H': [1],
          'P_W': [1],
          'P_C1': factor(T_C1),
          'P_C2': factor(T_C2),
          'P_F': factor(T_F)
      })

      for P_H, P_W, P_C1, P_C2, P_F in itertools.product(*list(params.values())):
        if use_winograd:
          P_COL = P_C1 * 4
          P_ROW = P_F * P_C2 * 9
        else:
          P_COL = P_H * P_W * P_C1
          P_ROW = P_F * P_C2 * 9

        yield [T_H, T_W, T_C1, T_C2, T_F, P_H, P_W, P_C1, P_C2, P_F, T_ROW, T_COL, P_ROW, P_COL]

  def candidates(self, block_type, use_winograd=False):
    if block_type == 'Bottleneck':
      return self.bottleneck_candidates(use_winograd=use_winograd)

  def explore(self, net: ConvNet, platform: Platform, use_winograd=False):
    predictor = BasicPredictor()

    results = []
    for cfg in self.candidates(use_winograd):
      params = DesignParams(T_H=cfg[0], T_W=cfg[1], T_C=cfg[2], T_F=cfg[3], P_H=cfg[4], P_W=cfg[5],
                            P_C=cfg[6], P_F=cfg[7], T_ROW=cfg[8], T_COL=cfg[9], P_ROW=cfg[10], P_COL=cfg[11])

      design = BasicDesign(params)
      res = design.resource
      BRAM, DSP, ALM = res.BRAM, res.DSP, res.ALM

      if BRAM > platform.max_BRAM or DSP > platform.max_DSP or ALM > platform.max_ALM:
        continue

      num_cycle, elapsed, GFLOPS, _, CTC = predictor.predict(net, design)

      results.append([BRAM, DSP, ALM, num_cycle, elapsed, GFLOPS, CTC, params])

    return np.array(results)
