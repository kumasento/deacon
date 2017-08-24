#!/bin/sh

PF=(1 2 4 8 16)
PC=(1 2 4 8 16)
BW=(32 16 8)
K=$1
echo "K = $K"

for pf in ${PF[@]}; do
    for pc in ${PC[@]}; do
        echo "PF = $pf PC = $pc"
        make build bitWidth=32 K=$K PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=32 K=$K PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=16 K=$K PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=16 K=$K PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=8  K=$K PF=$pf PC=$pc PK=1 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null &
        make build bitWidth=8  K=$K PF=$pf PC=$pc PK=2 MAXCOMPILER_BUILD_DIR=/mnt/data/scratch/rz3515/builds &> /dev/null
    done
done
