#!/usr/bin/env python

import caffe
import numpy as np
import os
import sys
import argparse

dir_path  = os.path.dirname(os.path.realpath(__file__))
test_path = os.path.join(dir_path, '..', '..', 'src', 'test')
test_data_path = os.path.join(dir_path, '..', 'test')

def parse_args():
  parser = argparse.ArgumentParser(description='Generate data file')
  parser.add_argument('test_name', help='Name of the test')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  print 'Running data generator'
  caffe.set_mode_cpu()
  
  test_name = args.test_name
  prototxt = os.path.join(test_path, test_name + '.prototxt')
  if not os.path.isfile(prototxt):
    print prototxt + ' is not a valid file path'
    exit(1)

  net = caffe.Net(prototxt, caffe.TEST)

  # fulfill each blob with random data
  # first take the input blob (named 'data')
  data_blob = net.blobs.get('data')
  if data_blob is None:
    print 'The network definition must contain the \'data\' blob'
    exit(1)

  data_blob.data[...] = np.asarray(
      np.random.random_sample(data_blob.data.shape), dtype=np.float32)

  data_dir = os.path.join(test_data_path, test_name)
  if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

  for layer_name, param in net.params.iteritems():
    for idx in range(len(param)):
      # fulfill each param in the layer
      param[idx].data[...] = np.asarray(
          np.random.random_sample(param[idx].data.shape), dtype=np.float32)
      print param[idx].data.shape

      file_path = os.path.join(data_dir, layer_name + '_param_' + str(idx) + '.bin')
      param[idx].data.tofile(file_path, format='%.10f')

  net.forward()

  # output the blob data
  for blob_name, blob in net.blobs.iteritems():
    file_path = os.path.join(data_dir, blob_name + '.bin')
    blob.data.tofile(file_path, format='%.10f')
    print blob.data.shape
