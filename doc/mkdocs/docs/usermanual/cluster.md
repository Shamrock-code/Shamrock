# Running on clusters

# CBP
On CBP machines, whether equipped with AMD or Nvidia GPUs, or without GPUs, Shamrock can be installed using your compiler of choice: either `intel-llvm` or `llvm+acpp`.

To set up the environment, use one of the following commands:

```bash
# For intel-llvm compiler
./env/new-env --machine cbp.intel-llvm --builddir build_cbp.intel-llvm

# For acpp compiler
./env/new-env --machine cbp.acpp --builddir build_cbp.acpp
```

For the acpp compiler, a generic installation (without architecture targeting) with `sscp` support is possible using the `--backend sscp` flag:

```bash
env/new-env --machine cbp.acpp --builddir build_cbp.acpp-sscp -- --backend sscp
```

# CBP (AMD GPU)

`git clone https://github.com/intel/llvm.git`

```bash
python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=../../dpcpp_compiler" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" --cmake-gen "Unix Makefiles"
```

```bash
cd build
make -j all libsycldevice install
```

# CBP (Nvidia GPU)

`git clone https://github.com/intel/llvm.git`

```bash
python3 buildbot/configure.py --cuda --cmake-opt="-DCMAKE_INSTALL_PREFIX=../../dpcpp_compiler" --cmake-gen "Unix Makefiles"
```

```bash
cd build
make -j all libsycldevice install
```


## Neowise
10 nodes, 10 cpus, 480 cores, 8 gpu per nodes


### Compiling OpenSYCL

Seem's to be broken due to gcc headers
```bash
module purge
module load llvm-amdgpu/5.2.0_gcc-10.4.0 rocm-cmake/5.2.0_gcc-10.4.0 rocm-opencl/5.2.0_gcc-10.4.0 rocm-openmp-extras/5.2.0_gcc-10.4.0 rocm-smi-lib/5.2.3_gcc-10.4.0 rocminfo/5.2.0_gcc-10.4.0 llvm
```

```bash
cd OpenSYCL

cmake \
  -DWITH_ROCM_BACKEND=ON \
  -DWITH_SSCP_COMPILER=OFF \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX=../OpenSYCL_comp .

make -j install
```

```bash
cd Shamrock
python3 buildbot/configure.py --gen make --tests --build release --outdir build --cxxpath ../OpenSYCL_comp --compiler opensycl --profile hip-gfx906
```
### Compiling dpcpp

```bash
module load hip
module load openmpi

```

```bash
python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=../../dpcpp_compiler" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" --cmake-gen "Unix Makefiles"
```

```bash
export DPCPP_HOME=$(pwd)/dpcpp_compiler
export PATH=$DPCPP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/lib:$LD_LIBRARY_PATH

tar -xvf Shamrock.tar.gz
cd Shamrock
python3 buildbot/configure.py --gen make --tests --build release --builddir dpcpp_rocm --cxxpath ../llvm/build --compiler intel_llvm --profile hip-gfx906 --cxxflags="--rocm-path=/opt/rocm"
cd dpcpp_rocm
make -j
```

Runing the code on 8 gpu per nodes
```bash
module load hip
module load openmpi
$(which mpirun) -machinefile $OAR_NODEFILE -npernode 8 -x PATH=~/dpcpp_compiler/bin:$PATH -x LD_LIBRARY_PATH=~/dpcpp_compiler/lib:$LD_LIBRARY_PATH ./shamrock --sycl-cfg auto:HIP --loglevel 1 --smi  --benchmark-mpi --rscript ../exemples/spherical_wave.py
```


```bash
module load hip
module load openmpi
$(which mpirun) -machinefile $OAR_NODEFILE -npernode 8 --mca pml ucx -x UCX_TLS=self,sm,rocm -x PATH=~/dpcpp_compiler/bin:$PATH -x LD_LIBRARY_PATH=~/dpcpp_compiler/lib:$LD_LIBRARY_PATH ./shamrock --sycl-cfg auto:HIP --loglevel 1 --smi  --benchmark-mpi --rscript ../exemples/spherical_wave.py
```




```
  359  oarsub -t exotic -p neowise -l gpu=1 -I
  360  oarsub -t exotic -p neowise -l gpu=16,walltime=0:30 -I
  361  oarsub -t exotic -p neowise -l gpu=16,walltime=0:30 -I
  362  oarsub -t exotic -p neowise -l gpu=32,walltime=0:30 -I
  363  oarsub -t exotic -p neowise -l gpu=64,walltime=0:30 -I
```
