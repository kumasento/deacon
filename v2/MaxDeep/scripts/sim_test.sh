#!/bin/bash

# This file contains simulation test scripts for several critical designs and parameters

sim_conv2d() {
  num_of_pipes=$1
  num_of_pumps=$2
  printf "CONV2D (3, %2d, %d)  " $num_of_pipes $num_of_pumps
  make -C ../build runsim \
    DESIGN_NAME=CONV2D \
    KERNEL_SIZE=3 \
    NUM_PIPES=$num_of_pipes \
    MULTI_PUMPING_FACTOR=$num_of_pumps \
    MAX_CONV_HEIGHT=32 \
    MAX_CONV_WIDTH=32 \
    MAX_CONV_NUM_CHANNELS=4 \
    MAX_CONV_NUM_FILTERS=64 \
    2>1 | grep "TEST "
}

echo -n "ONE_DIM_CONV (1) "
make -C ../build runsim DESIGN_NAME=ONE_DIM_CONV NUM_PIPES=1 | grep "TEST "
echo -n "ONE_DIM_CONV (2) "
make -C ../build runsim DESIGN_NAME=ONE_DIM_CONV NUM_PIPES=2 | grep "TEST "

for num_of_pumps in 1 2 4; do
  for num_of_pipes in 1 2 4 8 16 32 64; do
    if [ $num_of_pipes -ge $num_of_pumps ]; then
      sim_conv2d $num_of_pipes $num_of_pumps
    fi
  done
done
