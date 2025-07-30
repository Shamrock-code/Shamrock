# Using CUDA aware openMPI


## UCX setup
```bash
https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz
tar -xvf ucx-1.15.0.tar.gz

cd ucx-1.15.0
./configure --prefix=/opt/ucx_cuda --with-cuda=/usr/local/cuda

make -j8
sudo make install
```

## Openmpi setup

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz

tar -xvf openmpi-4.1.6.tar.gz

./configure --prefix=/opt/openmpi_cuda --with-cuda=/usr/local/cuda --with-ucx=/opt/ucx_cuda
make -j8
sudo make install
```

Finally export the path tp the newly built OpenMPI
```bash
export PATH=/opt/openmpi_cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/openmpi_cuda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Now if you build shamrock you should see this mpi used.

To check run : `cmake .` in the build folder, the begining of the outputs contain the path to the MPI in use.

Also when shamrock (or the tests) starts if MPI is CUDA aware during startup it should print :

```
-----------------------------------------------------
MPI status :
 - MPI & SYCL init: Ok
 - MPI CUDA-AWARE : Yes
 - MPI ROCM-AWARE : Unknown
 - MPI use Direct Comm : Yes
 - MPI use Direct Comm : Working
-----------------------------------------------------
```
