#!/bin/bash
#SBATCH --account=cad14954
#SBATCH --job-name=Shamrock
#SBATCH --constraint=MI250
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --output=n008.%A.out
#SBATCH --time=0:15:00
#
echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo
#
WORKDIR=$(pwd)
RSCRIPT=$(pwd)/runscript.py
#
SHAMROCK_PATH=$SCRATCHDIR/Shamrock/build
cd $SCRATCHDIR/Shamrock/build
#
source ./activate
#
export DUMP_PATH=$WORKDIR
#
ldd ./shamrock
#
chmod +x ./binding_script.sh
srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest -- \
    ./binding_script.sh ./shamrock --sycl-cfg auto:HIP --loglevel 1 --smi \
    --rscript $RSCRIPT
