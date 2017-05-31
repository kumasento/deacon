#!/bin/bash

# Run multiple builds of conv2d

run_conv2d_build() {
  num_chnl_pipes=$1
  num_fltr_pipes=$2
  freq=$3
  num_of_pumps=$4

  printf 'building CONV2D CHNL_PIPES=%3d FLTR_PIPES=%3d NUM_PUMPS=%d FREQ=%3d ...\n' \
    $num_chnl_pipes $num_fltr_pipes $num_of_pumps $freq

  make -C ../build build \
    DESIGN_NAME=CONV2D \
    BITWIDTH=16 \
    FREQ=$freq \
    NUM_CONV_CHNL_PIPES=$num_chnl_pipes \
    NUM_CONV_FLTR_PIPES=$num_fltr_pipes \
    MULTI_PUMPING_FACTOR=$num_of_pumps \
    KERNEL_SIZE=3 \
    MAX_CONV_HEIGHT=32 \
    MAX_CONV_WIDTH=32 \
    MAX_CONV_NUM_CHANNELS=512 \
    MAX_CONV_NUM_FILTERS=512 2>1 | grep "([0-9]*/[0-9]*)"
}

run_conv2d_parallel_pipes_builds() {
  run_conv2d_build 1 1  100 1 &
  run_conv2d_build 1 2  100 1 &
  run_conv2d_build 1 4  100 1 &
  run_conv2d_build 1 8  100 1
  run_conv2d_build 1 16 100 1 &
  run_conv2d_build 1 32 100 1

  run_conv2d_build 1 1  150 1 &
  run_conv2d_build 1 2  150 1 
  run_conv2d_build 1 4  150 1 &
  run_conv2d_build 1 8  150 1
  run_conv2d_build 1 16 150 1 &
  run_conv2d_build 1 32 150 1

  run_conv2d_build 1 1  200 1 &
  run_conv2d_build 1 2  200 1 
  run_conv2d_build 1 4  200 1 &
  run_conv2d_build 1 8  200 1
  run_conv2d_build 1 16 200 1 &
  run_conv2d_build 1 32 200 1
}

run_conv2d_multi_pumped_builds() {
  run_conv2d_build 1 2  100 2 &
  run_conv2d_build 1 4  100 2 
  run_conv2d_build 1 2  150 2 &
  run_conv2d_build 1 4  150 2 
  run_conv2d_build 1 2  200 2 &
  run_conv2d_build 1 4  200 2 
  run_conv2d_build 1 8  100 2 &
  run_conv2d_build 1 16 100 2 
  run_conv2d_build 1 32 100 2 &
  run_conv2d_build 1 64 100 2 
}

run_conv2d_large_builds() {
  run_conv2d_build 1  128 100 1 &
  run_conv2d_build 2  128 100 1 &
  run_conv2d_build 3  128 100 1 &
  run_conv2d_build 5  128 100 1 
  run_conv2d_build 10 128 100 1 &
  run_conv2d_build 15 128 100 1 &
  run_conv2d_build 30 128 100 1 
}

run_conv2d_very_large_builds() {
  run_conv2d_build 1 256 100 1
  run_conv2d_build 1 256 100 2
  run_conv2d_build 1 256 150 1
  run_conv2d_build 1 256 150 2
  run_conv2d_build 1 256 200 1
  run_conv2d_build 1 256 200 2
}

run_conv2d_chnl_builds() {
  run_conv2d_build 1 1 100 1 &
  run_conv2d_build 2 1 100 1 &
  run_conv2d_build 1 2 100 1 &
  run_conv2d_build 2 2 100 1 
  run_conv2d_build 1 2 100 2 &
  run_conv2d_build 2 2 100 2 
}
export -f run_conv2d_build

if [[ $1 = "MP" ]]; then
  run_conv2d_multi_pumped_builds
elif [[ $1 = "LG" ]]; then
  run_conv2d_large_builds
elif [[ $1 = "VERY_LG" ]]; then
  run_conv2d_very_large_builds
elif [[ $1 = "CHNL" ]]; then
  run_conv2d_chnl_builds
else
  run_conv2d_parallel_pipes_builds
fi
