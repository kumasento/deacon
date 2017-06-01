#!/bin/bash

DIRNAME=$(dirname $BASH_SOURCE)
source $DIRNAME/run_conv2d_sim.sh

run_conv2d_sim 1 1 32 | grep TEST
run_conv2d_sim 2 1 32 | grep TEST
run_conv2d_sim 2 2 32 | grep TEST

run_conv2d_sim 1 1 16 | grep TEST
run_conv2d_sim 2 1 16 | grep TEST
run_conv2d_sim 2 2 16 | grep TEST
