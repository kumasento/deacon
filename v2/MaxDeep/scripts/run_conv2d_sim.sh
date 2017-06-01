#!/bin/bash

run_conv2d_sim() {
  num_chnl_pipes=$1
  num_fltr_pipes=$2
  num_read_pipes=$3
  multi_pumping_factor=$4
  bitwidth=$5
  DEBUG=${DEBUG:-false}

  printf "SIM %d %d %d %d %d\n\n" $num_chnl_pipes $num_fltr_pipes $num_read_pipes $multi_pumping_factor $bitwidth

  make -C ../build runsim \
    DESIGN_NAME=CONV2D \
    KERNEL_SIZE=3 \
    NUM_CONV_CHNL_PIPES=$num_chnl_pipes \
    NUM_CONV_FLTR_PIPES=$num_fltr_pipes \
    NUM_PIPES=$num_read_pipes \
    MULTI_PUMPING_FACTOR=$multi_pumping_factor \
    MAX_CONV_NUM_CHANNELS=4 \
    MAX_CONV_NUM_FILTERS=24 \
    NUM_ITERS=1 \
    DEBUG=$DEBUG
    BITWIDTH=$bitwidth
}
export -f run_conv2d_sim
