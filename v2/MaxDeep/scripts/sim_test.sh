#!/bin/bash

# This file contains simulation test scripts for several critical designs and parameters

echo -n "ONE_DIM_CONV (1) "
make -C ../build runsim DESIGN_NAME=ONE_DIM_CONV NUM_PIPES=1 | grep "TEST "
echo -n "ONE_DIM_CONV (2) "
make -C ../build runsim DESIGN_NAME=ONE_DIM_CONV NUM_PIPES=2 | grep "TEST "
echo -n "CONV2D (4, 1)    "
make -C ../build runsim DESIGN_NAME=CONV2D KERNEL_SIZE=4 NUM_PIPES=1 2>1 | grep "TEST "
# echo -n "CONV2D (4, 2)    "
# make -C ../build runsim DESIGN_NAME=CONV2D KERNEL_SIZE=4 NUM_PIPES=2 2>1 | grep "TEST "
