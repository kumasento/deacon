#!/usr/bin/env python

from __future__ import print_function

class OneDimConv(object):
  def __init__(self, num_pipes, freq):
    self.num_pipes = num_pipes
    self.freq = freq

  def bandwidth_required(self):
    return self.num_pipes * 4. / (1. / (self.freq * 1e6)) * 1e-9

if __name__ == '__main__':
  print(OneDimConv(4, 100).bandwidth_required())
  print(OneDimConv(8, 100).bandwidth_required())
  print(OneDimConv(16, 100).bandwidth_required())
  print(OneDimConv(48, 100).bandwidth_required())
  print(OneDimConv(96, 100).bandwidth_required())
