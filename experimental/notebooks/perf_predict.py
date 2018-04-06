from net import ConvNet
from layer import ConvLayer
from design import Design


class Predictor(object):
  def __init__(self):
    pass

  def predict(self, net):
    pass


class BasicPredictor(Predictor):
  def predict(self, net: ConvNet, design: Design, freq=200):
    num_cycles = design.run(net)
    num_secs = num_cycles * 1. / (freq * 1e6)
    gflops = net.num_ops * 1e-9 / num_secs

    return num_cycles, num_secs, gflops
