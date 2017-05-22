#!/bin/bash
# This script builds ONE_DIM_CONV for further experiments

function run_build {
  echo "building $1 design with frequency $2MHz num_pipes $3 ..."
  make -C ../build build \
    DESIGN_NAME=$1 \
    FREQ=$2 \
    NUM_PIPES=$3 \
    ONE_DIM_CONV_WINDOW_WIDTH=$4 \
    | grep "([0-9]*/[0-9]*)"
}

function run_builds {
  name=$1
  run_build $name 100 1   3 &
  run_build $name 100 2   3 &
  run_build $name 100 4   3 &
  run_build $name 100 8   3
  run_build $name 100 16  3 &
  run_build $name 100 48  3 &
  run_build $name 100 96  3 &
  run_build $name 100 192 3

  run_build $name 150 1 3 &
  run_build $name 150 2 3 
  run_build $name 150 4 3 &
  run_build $name 150 8 3

  run_build $name 200 1 3
  run_build $name 200 2 3 
  run_build $name 200 4 3
  run_build $name 200 8 3
}

run_builds ONE_DIM_CONV
