#!/usr/bin/env python

from __future__ import print_function

import os
import csv

BUILD_PARAMS = [
  [ 9,  32 ], [ 16, 32 ], [ 25, 32 ], [ 36, 32 ], [ 49, 32 ],
  [ 9,  16 ], [ 16, 16 ], [ 25, 16 ], [ 36, 16 ], [ 49, 16 ],
  [ 9,  8  ], [ 16, 8  ], [ 25, 8  ], [ 36, 8  ], [ 49, 8  ],
  [ 9,  4  ], [ 16, 4  ], [ 25, 4  ], [ 36, 4  ], [ 49, 4  ],
]

if __name__ == '__main__':

  with open('data/dotprod.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=' ')
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

