#!/bin/sh

PF=(1 2 4 8 16)
PC=(1 2 4 8 16)
BW=(32 16 8)

for pf in ${PF[@]}; do
    for pc in ${PC[@]}; do
        make build bitWidth=32 PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=32 PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=16 PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=16 PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=8  PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=8  PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null
    done
done
