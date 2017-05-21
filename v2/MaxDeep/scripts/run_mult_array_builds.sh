#!/bin/bash

freq_list=(100 150 200)
num_pipe_list=(1 2 4 8)

function run_build {
  echo "building $1 design with frequency $2MHz num_pipes $3 ..."
  make -C ../build build DESIGN_NAME=$1 FREQ=$2 NUM_PIPES=$3 | grep "([0-9]*/[0-9]*)" 
}

function run_builds {
  name=$1
  run_build $name 100 1 &
  run_build $name 100 2 &
  run_build $name 100 4 &
  run_build $name 100 8

  run_build $name 150 1 &
  run_build $name 150 2 
  run_build $name 150 4 &
  run_build $name 150 8

  run_build $name 200 1
  run_build $name 200 2 
  run_build $name 200 4
  run_build $name 200 8
}

run_builds $1
