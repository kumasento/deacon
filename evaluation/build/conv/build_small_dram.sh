#!/bin/sh

PF=($1)
PC=(1 2 4 8 16)

for pf in ${PF[@]}; do
    for pc in ${PC[@]}; do
        make build USE_DRAM=true PF=$pf PC=$pc PK=1 bitWidth=32 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &
        make build USE_DRAM=true PF=$pf PC=$pc PK=2 bitWidth=32 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds
    done
done
