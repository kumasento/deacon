#!/bin/bash

# Run multiple builds of conv2d

run_conv2d_build() {
  num_of_pipes=$1
  freq=$2
  num_of_pumps=$3

  printf 'building CONV2D NUM_PIPES=%3d NUM_PUMPS=%d FREQ=%3d ...\n' $num_of_pipes $num_of_pumps $freq
  make -C ../build build \
    DESIGN_NAME=CONV2D \
    BITWIDTH=16 \
    FREQ=$freq \
    NUM_PIPES=$num_of_pipes \
    MULTI_PUMPING_FACTOR=$num_of_pumps \
    KERNEL_SIZE=3 \
    MAX_CONV_HEIGHT=32 \
    MAX_CONV_WIDTH=32 \
    MAX_CONV_NUM_CHANNELS=512 \
    MAX_CONV_NUM_FILTERS=512 2>1 | grep "([0-9]*/[0-9]*)"
}

run_conv2d_parallel_pipes_builds() {
  run_conv2d_build 1  100 1 &
  run_conv2d_build 2  100 1 &
  run_conv2d_build 4  100 1 &
  run_conv2d_build 8  100 1
  run_conv2d_build 16 100 1 &
  run_conv2d_build 32 100 1

  run_conv2d_build 1  150 1 &
  run_conv2d_build 2  150 1 
  run_conv2d_build 4  150 1 &
  run_conv2d_build 8  150 1
  run_conv2d_build 16 150 1 &
  run_conv2d_build 32 150 1

  run_conv2d_build 1  200 1 &
  run_conv2d_build 2  200 1 
  run_conv2d_build 4  200 1 &
  run_conv2d_build 8  200 1
  run_conv2d_build 16 200 1 &
  run_conv2d_build 32 200 1
}

run_conv2d_multi_pumped_builds() {
  run_conv2d_build 2  100 2 &
  run_conv2d_build 4  100 2 
  run_conv2d_build 2  150 2 &
  run_conv2d_build 4  150 2 
  run_conv2d_build 2  200 2 &
  run_conv2d_build 4  200 2 
  run_conv2d_build 8  100 2 &
  run_conv2d_build 16 100 2 
  run_conv2d_build 32 100 2 &
  run_conv2d_build 64 100 2 
}

run_conv2d_large_builds() {
  run_conv2d_build 128 100 1
  run_conv2d_build 128 100 2
  run_conv2d_build 128 150 1
  run_conv2d_build 128 150 2
  run_conv2d_build 128 200 1
  run_conv2d_build 128 200 2
}

if [[ $1 = "MP" ]]; then
  run_conv2d_multi_pumped_builds
elif [[ $1 = "LG" ]]; then
  run_conv2d_large_builds
else
  run_conv2d_parallel_pipes_builds
fi
