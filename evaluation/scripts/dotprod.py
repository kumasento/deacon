#!/usr/bin/env python

from __future__ import print_function

import os
import csv

VEC_SIZE_LIST = [ 9, 16, 25, 36, 49, 64, 81, 100, 121 ]

BIT_WIDTH_LIST = [ 32, 16, 8, 4 ]

BUILD_PARAMS = [ ]

for b in BIT_WIDTH_LIST:
  for v in VEC_SIZE_LIST:
    BUILD_PARAMS.append([v, b])

if __name__ == '__main__':

  with open('data/dotprod.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['bit_width', 'vec_size', 'LUT', 'FF', 'BRAM', 'DSP'])

    for build_param in BUILD_PARAMS:
      bit_width = build_param[1]
      vec_size = build_param[0]

      file_path = os.path.join(
          '/mnt/data/scratch/rz3515/builds',
          'DotProd_MAIA_DFE_b{0}_n{1}'.format(bit_width, vec_size),
          'src_annotated_DOT_PROD_KERNEL/dotprod/DotProdKernel.maxja')
      print(file_path)

      with open(file_path, 'r') as f:
        lines = f.readlines()
        line = lines[36]
        line = line.strip()
        usage = line.split()[:4]

      csvwriter.writerow([bit_width, vec_size] + usage)

