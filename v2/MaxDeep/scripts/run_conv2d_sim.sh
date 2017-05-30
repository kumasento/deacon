#!/bin/bash

run_conv2d_sim() {
  num_pipes=$1
  multi_pumping_factor=$2
  bitwidth=$3

  make -C ../build runsim \
    DESIGN_NAME=CONV2D \
    KERNEL_SIZE=3 \
    NUM_PIPES=$num_pipes \
    MULTI_PUMPING_FACTOR=$multi_pumping_factor \
    MAX_CONV_NUM_CHANNELS=32 \
    MAX_CONV_NUM_FILTERS=32 \
    NUM_ITERS=1 \
    BITWIDTH=$bitwidth
}
export -f run_conv2d_sim
