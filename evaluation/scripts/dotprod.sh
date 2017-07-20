#!/bin/bash

# Run dotprod evaluation
DOT_PROD_BUILD_ROOT=$(dirname ${BASH_SOURCE[0]})/../build/dotprod
make -C ${DOT_PROD_BUILD_ROOT} build BIT_WIDTH=32
make -C ${DOT_PROD_BUILD_ROOT} build BIT_WIDTH=16
make -C ${DOT_PROD_BUILD_ROOT} build BIT_WIDTH=8
make -C ${DOT_PROD_BUILD_ROOT} build BIT_WIDTH=4
