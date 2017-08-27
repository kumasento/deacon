#!/bin/sh

PF=(1 2 4 8 16)
PC=(1 2 4 8 16)

for pf in ${PF[@]}; do
    for pc in ${PC[@]}; do
        make build PF=$pf PC=$pc PK=1 bitWidth=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &
        make build PF=$pf PC=$pc PK=2 bitWidth=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds
    done
done
