# Running on cluster

# LLVM local install on cluster

Make a shallow clone of llvm :
```
git clone --depth 1 https://github.com/llvm/llvm-project.git -b release/17.x
cd llvm-project
```

```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.1/llvm-project-17.0.1.src.tar.xz
tar -xvf llvm-project-17.0.1.src.tar.xz
cd llvm-project-17.0.1.src
```

configure it :
```
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=all -DCMAKE_INSTALL_PREFIX=...instal loc.../llvm-17.x-local -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
```

compile it : 
```
make -j install
```


```
cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_INSTALL_PREFIX=/local/tdavidcl/llvm-17.x-local -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
```

# CBP (AMD GPU)

`git clone https://github.com/intel/llvm.git`

```
python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=../../dpcpp_compiler" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" --cmake-gen "Unix Makefiles"
```

```
cd build
make -j all libsycldevice install
```

# CBP (Nvidia GPU)

`git clone https://github.com/intel/llvm.git`

```
python3 buildbot/configure.py --cuda --cmake-opt="-DCMAKE_INSTALL_PREFIX=../../dpcpp_compiler" --cmake-gen "Unix Makefiles"
```

```
cd build
make -j all libsycldevice install
```


## Neowise
10 nodes, 10 cpus, 480 cores, 8 gpu per nodes


### Compiling OpenSYCL

Seem's to be broken due to gcc headers  
```
module purge
module load llvm-amdgpu/5.2.0_gcc-10.4.0 rocm-cmake/5.2.0_gcc-10.4.0 rocm-opencl/5.2.0_gcc-10.4.0 rocm-openmp-extras/5.2.0_gcc-10.4.0 rocm-smi-lib/5.2.3_gcc-10.4.0 rocminfo/5.2.0_gcc-10.4.0 llvm
```

```
cd OpenSYCL

cmake \
  -DWITH_ROCM_BACKEND=ON \
  -DWITH_SSCP_COMPILER=OFF \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX=../OpenSYCL_comp .

make -j install
```

```
cd Shamrock
python3 buildbot/configure.py --gen make --tests --build release --outdir build --cxxpath ../OpenSYCL_comp --compiler opensycl --profile hip-gfx906
```
### Compiling dpcpp

```
python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=../../dpcpp_compiler" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" --cmake-gen "Unix Makefiles"
```

```bash
export DPCPP_HOME=$(pwd)/dpcpp_compiler
export PATH=$DPCPP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/lib:$LD_LIBRARY_PATH

tar -xvf Shamrock.tar.gz 
cd Shamrock
python3 buildbot/configure.py --gen make --tests --build release --outdir dpcpp_rocm --cxxpath ../llvm/build --compiler dpcpp --profile hip-gfx906 --cxxflags="--rocm-path=/opt/rocm"
cd dpcpp_rocm
make -j
```

Runing the code on 8 gpu per nodes
```bash
mpirun -machinefile $OAR_NODEFILE -npernode 8 -x PATH=~/dpcpp_compiler/bin:$PATH -x LD_LIBRARY_PATH=~/dpcpp_compiler/lib:$LD_LIBRARY_PATH ./shamrock --sycl-cfg auto:HIP --loglevel 1 --sycl-ls-map  --rscript ../exemples/spherical_wave.py
```
