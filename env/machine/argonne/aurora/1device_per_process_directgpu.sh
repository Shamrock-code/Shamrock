#!/bin/bash

# If you want shamrock to think of mpi as slurm like srun
#export LOCAL_RANK=$PALS_LOCAL_RANKID

# Each process see a single device
export ZE_AFFINITY_MASK=$PALS_LOCAL_RANKID

# Enable direct GPU comm (do not forget to use --force-dgpu-on)
export MPIR_CVAR_ENABLE_GPU=1

exec "$@"
