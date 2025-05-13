#!/bin/bash

# If you want shamrock to think of mpi as slurm like srun
#export LOCAL_RANK=$PALS_LOCAL_RANKID

# Each process see a single device
export ZE_AFFINITY_MASK=$LOCAL_RANK

# See each tile instead of the full device
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

exec "$@"
