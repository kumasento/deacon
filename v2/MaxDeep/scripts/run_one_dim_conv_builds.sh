#!/bin/bash
# This script builds ONE_DIM_CONV for further experiments

function run_build {
  echo "building $1 design with frequency $2MHz num_pipes $3 ..."
  make -C ../build build \
    DESIGN_NAME=$1 \
    FREQ=$2 \
    NUM_PIPES=$3 \
    ONE_DIM_CONV_WINDOW_WIDTH=$4 \
    MULTI_PUMPING_FACTOR=$5 \
    | grep "([0-9]*/[0-9]*)"
}

function run_builds {
  name=$1
 
  run_build $name 100 1 3 1 &
  run_build $name 100 2 3 1 &
  run_build $name 100 4 3 1 &
  run_build $name 100 8 3 1

  run_build $name 100 2 3 2
  run_build $name 100 4 3 2
  run_build $name 100 8 3 2
  run_build $name 100 16 3 2
                          
  run_build $name 150 1 3 1 &
  run_build $name 150 2 3 1
  run_build $name 150 4 3 1 &
  run_build $name 150 8 3 1

  run_build $name 150 2 3 2
  run_build $name 150 4 3 2 
  run_build $name 150 8 3 2
  run_build $name 150 16 3 2
                          
  run_build $name 200 1 3 1
  run_build $name 200 2 3 1
  run_build $name 200 4 3 1
  run_build $name 200 8 3 1
}

run_builds ONE_DIM_CONV
