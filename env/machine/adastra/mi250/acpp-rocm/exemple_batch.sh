#!/bin/bash
#SBATCH --account=<project account>
#SBATCH --job-name=<jobname>
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.out   # Same file for both!


. /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module list

export SYCL_DEVICE_ALLOCATOR=aligned

WORKDIR=<path-to-Shamrock-build-directory>

export SHAM_MAX_ALLOC_SIZE=4294967296

. ./activate

srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpus-per-task=1 --gpu-bind=closest -- ./shamrock --smi-full --sycl-cfg auto:HIP --force-dgpu-off --loglevel 1 --rscript runscript.py
