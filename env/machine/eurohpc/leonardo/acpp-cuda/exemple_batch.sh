#!/bin/bash -l
#SBATCH --job-name=<jobname>
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg #debug qos
#SBATCH --output=example.%j.out
#SBATCH --error=example.%j.err
#SBATCH --account=<account>

BUILD=<path-to-build>
source $BUILD/activate                       # same modules as the build

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ACPP_DEBUG_LEVEL=0

echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo

srun --ntasks-per-node=4 --cpus-per-task=8 --threads-per-core=1 '"$BUILD"'/shamrock --sycl-cfg auto:CUDA --loglevel 1 --smi --rscript runscript.py
