#!/bin/bash

function run_build {
  echo "building $1 design with frequency $2MHz ..."
  make -C ../build build DESIGN_NAME=$1 FREQ=$2 1&>/dev/null
}

function run_builds {
  run_build LOOPBACK 100 &
  run_build LOOPBACK 150
  run_build LOOPBACK 200 &
  run_build LOOPBACK 250
  run_build LOOPBACK 300 &
  run_build LOOPBACK 350 
}

run_builds
