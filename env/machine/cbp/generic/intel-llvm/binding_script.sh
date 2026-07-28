#!/bin/bash

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

case $OMPI_COMM_WORLD_LOCAL_RANK in
[0]) cpus=0-15 ;;
[1]) cpus=16-31 ;;
[2]) cpus=32-47 ;;
[3]) cpus=48-63 ;;
esac

# To bind cores as well
# echo "Process $OMPI_COMM_WORLD_LOCAL_RANK on cpus $cpus and GPU $CUDA_VISIBLE_DEVICES"
# numactl --physcpubind=$cpus $@

# Only bind devices and let linux do the rest (probably best honestly)
echo "Process $OMPI_COMM_WORLD_LOCAL_RANK on GPU $CUDA_VISIBLE_DEVICES"
$@
