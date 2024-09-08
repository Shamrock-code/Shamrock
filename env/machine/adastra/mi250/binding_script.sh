#!/bin/bash

set -eu

LOCAL_RANK_INDEX="${SLURM_LOCALID}"
LOCAL_RANK_COUNT="${SLURM_NTASKS_PER_NODE}"

function Adastra_MI250_8TasksWith8ThreadsAnd1GPU() {
    AFFINITY_NUMACTL=('48-55' '56-63' '16-23' '24-31' '0-7' '8-15' '32-39' '40-47')
    AFFINITY_GPU=('0' '1' '2' '3' '4' '5' '6' '7')
    export MPICH_OFI_NIC_POLICY=NUMA
}

Adastra_MI250_8TasksWith8ThreadsAnd1GPU

CPU_SET="${AFFINITY_NUMACTL[$((${LOCAL_RANK_INDEX} % ${#AFFINITY_NUMACTL[@]}))]}"
if [ ! -z ${AFFINITY_GPU+x} ]; then
    GPU_SET="${AFFINITY_GPU[$((${LOCAL_RANK_INDEX} % ${#AFFINITY_GPU[@]}))]}"
    export ROCR_VISIBLE_DEVICES="${GPU_SET}"
fi
exec numactl --localalloc --physcpubind="${CPU_SET}" -- "${@}"
