#!/bin/bash

run_conv2d_sim() {
  num_chnl_pipes=$1
  num_fltr_pipes=$2
  multi_pumping_factor=$3
  bitwidth=$4
  DEBUG=${DEBUG:-false}

  printf "SIM %d %d %d %d\n\n" $num_chnl_pipes $num_fltr_pipes $multi_pumping_factor $bitwidth

  make -C ../build runsim \
    DESIGN_NAME=CONV2D \
    KERNEL_SIZE=3 \
    NUM_CONV_CHNL_PIPES=$num_chnl_pipes \
    NUM_CONV_FLTR_PIPES=$num_fltr_pipes \
    MULTI_PUMPING_FACTOR=$multi_pumping_factor \
    MAX_CONV_NUM_CHANNELS=32 \
    MAX_CONV_NUM_FILTERS=32 \
    NUM_ITERS=1 \
    DEBUG=$DEBUG
    BITWIDTH=$bitwidth
}
export -f run_conv2d_sim
