#!/bin/bash

function run_build {
  echo "building $1 design with frequency $2MHz ..."
  make -C ../build build DESIGN_NAME=$1 FREQ=$2 1&>/dev/null
}

function run_builds {
  run_build $1 100
  run_build $1 150
  run_build $1 200
}

run_builds $1
