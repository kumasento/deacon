#!/bin/sh

PF=(1 2 4 8 16)
PC=(1 2 4 8 16)
BW=(32 16 8)

for bw in ${BW[@]}; do
    for pf in ${PF[@]}; do
        for pc in ${PC[@]}; do
            make build bitWidth=$bw PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &
            make build bitWidth=$bw PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds
        done
    done
done
