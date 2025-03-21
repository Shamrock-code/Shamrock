#!/bin/bash -l
#SBATCH --job-name=examplejob   # Job name
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
#SBATCH --partition=standard-g  # partition name
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --time=00:10:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001405  # Project for billing
#
echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo
#
SHAMROCK_PATH=/scratch/project_465001405/Shamrock_scalling/Shamrock
cd $SHAMROCK_PATH/build_acpp
#
RSCRIPT=$SHAMROCK_PATH/exemples/sedov_scale_test_updated.py
#
source activate
#
ldd ./shamrock

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

export MPICH_GPU_SUPPORT_ENABLED=1
export ACPP_DEBUG_LEVEL=0

#
srun --cpu-bind=${CPU_BIND} -- \
    ./select_gpu ./shamrock --force-dgpu-on --sycl-cfg auto:HIP --loglevel 1 --sycl-ls-map \
    --rscript $RSCRIPT
