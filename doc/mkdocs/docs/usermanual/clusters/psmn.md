# PSMN guide

Sorry if this is complicated but the PSMN does not really help the process, sadly ...
It is definitely harder than on most clusters out there

## Getting a copy of shamrock on the PSMN

First clone the Shamrock repository :
```bash
git clone --recurse-submodules git@github.com:<github username>/Shamrock.git
```

then you need to upload it to the PSMN, to do so first tar it, copy it to the PSMN, and then untar.
```bash
tar -cvf Shamrock.tar.gz Shamrock
scp Shamrock.tar.gz tdavidcl@allo-psmn:/home/<psmn username>
```

Now log on the psmn :
```bash
ssh <psmn username>@allo-psmn
```
And untar shamrock :
```bash
tar -xvf Shamrock.tar.gz
```

## Setup of the enviroment

Ok now we are getting into the tricky part, there is multiple steps :

 - Compiler toolchain setup (LLVM 17)
 - boost setup
 - Adaptive Cpp setup
 - Shamrock setup

First log on a compilation node (ex : `ssh s92node0` for cascade lake) :

```bash
ssh <compilation node>
```


### Loading the modules

```bash

module use /applis/PSMN/debian11/Cascade/modules/all

module load GCC/11.2.0
module load GCCcore/11.2.0
module load CMake/3.22.1-GCCcore-11.2.0
module load Boost/1.77.0-GCC-11.2.0

```

### LLVM 17

Trust me bro moment, just run this :
```bash
cd $HOME

wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.1/llvm-project-17.0.1.src.tar.xz
tar -xvf llvm-project-17.0.1.src.tar.xz

cd $HOME/llvm-project-17.0.1.src
mkdir build
cd $HOME/llvm-project-17.0.1.src/build
cmake \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;clang-tools-extra;openmp;polly;libc" \
    -DLLVM_ENABLE_RUNTIMES="libc;libcxx;libcxxabi" \
    -DCMAKE_INSTALL_PREFIX="$HOME/llvm-17.x-local" \
    -DCMAKE_BUILD_TYPE=Release \
    -G "Unix Makefiles" \
    ../llvm

make -j install
```

### ShamrockWorkspace

```bash
git clone https://github.com/Shamrock-code/ShamrockWorkspace.git
```

cd into it :
```bash
cd ShamrockWorkspace
```

activate it to register some script binary in the path :
```bash
source ./activate
```

then run this :
```bash
cd $HOME/ShamrockWorkspace/sycl_compiler_gits
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd $HOME/ShamrockWorkspace/sycl_compiler_gits/AdaptiveCpp

cmake \
    -DBoost_USE_STATIC_LIBS=on \
    -DCLANG_EXECUTABLE_PATH=$HOME/llvm-17.x-local/bin/clang++ \
    -DLLVM_DIR=$HOME/llvm-17.x-local/lib/cmake/llvm \
    -DWITH_SSCP_COMPILER=OFF -DWITH_OPENCL_BACKEND=OFF \
    -DWITH_LEVEL_ZERO_BACKEND=OFF \
    -DCMAKE_INSTALL_PREFIX=$HOME/ShamrockWorkspace/sycl_compilers/acpp \
    -B build \
    .

cd build
make -j install
```

Then go back to the main folder
```bash
cd $HOME/ShamrockWorkspace
```

Now that Acpp is compiled rerun activate
```bash
source ./activate
```

```bash
mv $HOME/Shamrock $HOME/ShamrockWorkspace
```

### Configure shamrock

Because of the PSMN weirdness we have to use a weird config :
```bash
python3 Shamrock/buildbot/configure.py --gen make --tests --build release \
    --builddir Shamrock/build_config/acpp_omp_release \
    --cxxpath $WORKSPACEDIR/sycl_compilers/acpp \
    --compiler acpp \
    --profile omp \
    --cxxflags="-L/applis/PSMN/debian11/Lake/software/GCCcore/11.2.0/lib64"
```

### Compiling shamrock
```bash
cd $HOME/ShamrockWorkspace/Shamrock/build_config/acpp_omp_release
make -j
```

### Try running shamrock and the tests

export omp path :

```bash
export LD_LIBRARY_PATH=$HOME/llvm-17.x-local/lib:$LD_LIBRARY_PATH
```

run the tests :
```bash
./shamrock_test --sycl-cfg 0:0
```


## Slurm scripts :

Slurm script exemple :

```bash linenums="1" title="slurm_script"
#!/bin/bash
#SBATCH --job-name=ScallingTests_Shamrock
#SBATCH -o ./%x.%j.%N.out           # output file
#SBATCH -e ./%x.%j.%N.err           # errors file
#SBATCH -p Cascade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # number of MPI processes per node
#SBATCH --cpus-per-task=48          # number of OpenMP threads per MPI process
#SBATCH --time=0-00:10:00           # day-hours:minutes:seconds
#SBATCH --mail-user=timothee.david--cleris@ens-lyon.fr
#SBATCH --mail-type=BEGIN,END,FAIL
#
echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo
#
cd $HOME/ShamrockWorkspace/Shamrock/build_config/acpp_omp_release
#
module use /applis/PSMN/debian11/Cascade/modules/all
module load GCC/11.2.0
module load GCCcore/11.2.0
module load CMake/3.22.1-GCCcore-11.2.0
module load Boost/1.77.0-GCC-11.2.0
module load OpenMPI/4.1.1-GCC-11.2.0
#
mpirun --bind-to socket -npernode 2 \
    -x LD_LIBRARY_PATH=$HOME/llvm-17.x-local/lib:$LD_LIBRARY_PATH \
    -x OMP_NUM_THREADS=96 \
    -x ACPP_DEBUG_LEVEL=0 \
    ./shamrock --sycl-cfg 0:0 --loglevel 1 --smi \
    --rscript ../../exemples/sedov_scale_test.py
```

write it in a file and then do `sbatch <slurm script>`
