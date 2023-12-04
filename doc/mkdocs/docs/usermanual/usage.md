# Shamrock usage

```
executable : ./shamrock 

Usage : 
--sycl-ls                        : list available devices 
--sycl-ls-map                    : list available devices & list of queue bindings 
--benchmark-mpi                  : micro benchmark for MPI 
--sycl-cfg      (idcomp:idalt)   : specify the compute & alt queue index 
--loglevel      (logvalue)       : specify a log level 
--nocolor                        : disable colored ouput 
--rscript       (filepath)       : run shamrock with python runscirpt 
--ipython                        : run shamrock in Ipython mode 
--help                           : show this message 
```

## Running on multiple nodes

For exemple on a cluster using OARsub the command to run 

```sh
mpirun -machinefile $OAR_NODEFILE --bind-to socket -npernode 2 sh runscript.sh
```

Here `mpirun` is the standard command to start a program using MPI (to start multiple process that can work together using MPI library). `-machinefile` is used to specify the layout of the nodes that will be used by MPI (if you have reserved 10 nodes for exemple this is the list of nodes available). Starting from OpenMPI 4 `--bind-to` specify the list of CPU cores that are used by each process started by MPI, here `--bind-to socket` tell MPI that each process should be attach to a socket. On the machine used is the exemple there are 2 CPUs per nodes therefor to start one process per CPU one should use `--bind-to socket -npernode 2`, where a socket refer to the "slot" where the CPU is installed (2 on this system). Here `sh runscript.sh` is just the command launched by each process here runscript is : 
```sh
export ACPP_DEBUG_LEVEL=0 
export LD_LIBRARY_PATH=/grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/llvm-13.0.1-i53qugtbmlvnfi6tppnc7bresushxg2j/lib:$LD_LIBRARY_PATH 
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=32

./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ../exemples/spherical_wave.py
```

Sadly hardware and software may differ greatly from clusters to clusters see [Cluster](cluster.md) for details on known clusters.