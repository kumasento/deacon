#!/bin/bash
# Removing useless scratch files in the build directories

BUILDS_ROOT=/mnt/data/scratch/rz3515/builds

for dir in ${BUILDS_ROOT}/*; do
  if [ -d ${dir} ]; then
    echo "Entering ${dir} ..."
    if [ -d ${dir}/scratch ]; then
      echo "Cleaning ${dir}/scratch ..."
      rm -rf ${dir}/scratch
    fi
  fi
done
