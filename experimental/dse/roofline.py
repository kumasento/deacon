#!/usr/bin/env python
"""
This module tries to figure out the roofline modelling.

The idea of using roofline models is originated from the FPGA '15 paper:
    Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks

"""

import abc
import math
import itertools

import numpy as np
import matplotlib.pyplot as plt

NUM_BITS_PER_BRAM = 20 * 1024 # M20K

MAX_NUM_BRAM = 2567
MAX_NUM_DSP = 3926

class RooflineModel(abc.ABC):
    @abc.abstractproperty
    def name(self):
        pass

    def CTC(self, **kwargs):
        """ The attainable performance.  """
        return float(self.total_num_ops(**kwargs)) / self.total_external_data_access(**kwargs)

    @abc.abstractmethod
    def GFLOPS(self, **kwargs):
        pass

    @abc.abstractmethod
    def total_num_ops(self, **kwargs):
        """ Total number of operations. """
        pass

    @abc.abstractmethod
    def total_external_data_access(self, **kwargs):
        """ Total number of external data access """
        pass

    @abc.abstractmethod
    def BRAM(self, **kwargs):
        pass

    @abc.abstractmethod
    def DSP(self, **kwargs):
        pass


class StandardConvLayerRooflineModel(RooflineModel):
    """
    Roofline model of the standard convolution layer
    hardware design.
    """
    
    def __init__(self, H, W, C, F, K, FREQ=150):
        self.H = H
        self.W = W
        self.C = C
        self.F = F
        self.K = K
        self.FREQ = FREQ

    @property
    def name(self):
        return "Standard Convolution"

    def total_num_ops(self, **kwargs):
        return 2 * self.H * self.W * self.C * self.F * self.K * self.K

    def total_external_data_access(self, **kwargs):
        NUM_BYTES = 4
        B_in = NUM_BYTES * (kwargs['T_H'] + self.K - 1) * (kwargs['T_W'] + self.K - 1) * self.C
        B_wgt = NUM_BYTES * self.K * self.K * self.F * self.C
        B_out = NUM_BYTES * kwargs['T_H'] * kwargs['T_W'] * self.F

        N_in = math.ceil(float(self.H) / kwargs['T_H']) * math.ceil(float(self.W) / kwargs['T_W'])
        N_wgt = N_in
        N_out = N_in

        return B_in * N_in + B_wgt * N_wgt + B_out * N_out

    def GFLOPS(self, **kwargs):
        num_cycles = (math.ceil(float(self.H) / kwargs['T_H']) * math.ceil(float(self.W) / kwargs['T_W']) *
                      kwargs['T_H'] * kwargs['T_W'] *
                      math.ceil(float(self.C) / kwargs['P_C']) * math.ceil(float(self.F) / kwargs['P_F']))
        num_secs = num_cycles * 1. / self.FREQ * 1e-6

        return self.total_num_ops(**kwargs) / num_secs * 1e-9

    def BRAM(self, **kwargs):
        num_bits_in_buf = (kwargs['T_H'] + self.K - 1) * (kwargs['T_W'] + self.K - 1) * self.C
        num_bits_line_buf = kwargs['P_C'] * self.K * kwargs['T_W']
        num_bits_out_buf = kwargs['T_H'] * kwargs['T_W'] * kwargs['P_F']

        num_bits = num_bits_in_buf + num_bits_line_buf + num_bits_out_buf
        num_bram = math.ceil(float(num_bits) / NUM_BITS_PER_BRAM)

        return num_bram

    def DSP(self, **kwargs):
        return kwargs['P_C'] * kwargs['P_F'] * self.K * self.K * 2
        
        
def plot():
    H, W, C, F, K = 224, 224, 3, 64, 3
    roofline_models = [
        StandardConvLayerRooflineModel(H, W, C, F, K)
    ]

    T_Hs = [4, 8, 16, 32, 48, 64]
    T_Ws = [4, 8, 16, 32, 48, 64]
    P_Cs = range(1, C + 1)
    P_Fs = range(1, F + 1)

    candidates = list(itertools.product(T_Hs, T_Ws, P_Cs, P_Fs))
    print('Number of candidates:', len(candidates))

    ctc_list, gflops_list = [], []

    xs = np.linspace(0, 15, num=100)

    compute_ys = [MAX_NUM_DSP / 2 / (1. / (200 * 1e6)) * 1e-9] * 100
    bandwidth_ys = xs * 38

    for model in roofline_models:
        for T_H, T_W, P_C, P_F in candidates:
            params = {
                'T_H': T_H,
                'T_W': T_W,
                'P_C': P_C,
                'P_F': P_F
            }

            bram, dsp = model.BRAM(**params), model.DSP(**params)
            if bram >= MAX_NUM_BRAM or dsp >= MAX_NUM_DSP:
                continue

            ctc, gflops = model.CTC(**params), model.GFLOPS(**params)

            ctc_list.append(ctc)
            gflops_list.append(gflops)

        plt.scatter(ctc_list, gflops_list, s=1.)
        plt.xlabel('Computation to Communication Ratio (FLOPS / byte)')
        plt.ylabel('Performance (GFLOPS)')
        plt.plot(xs, compute_ys)
        plt.plot(xs, bandwidth_ys)
        plt.savefig('%s.png' % model.name)
        plt.show()

if __name__ == '__main__':
    plot()

