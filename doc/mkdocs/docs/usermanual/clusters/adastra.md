# Adastra MI250X setup (LLVM)

## Compiler setup

### Compiling the compiler

To setup LLVM on adastra :
```bash
module purge

module load PrgEnv-amd
module load cray-python
module load CCE-GPU-2.1.0
module load rocm/5.7.1 # 5.5.1 -> 5.7.1

# to get cmake and ninja
pip3 install -U cmake ninja
export PATH=$HOMEDIR/.local/bin:$PATH

# do everything in scratch
cd $SCRATCHDIR

# The directory to use for the build
WORKDIR=$(pwd)

# get a shallow clone
git clone --depth 1 https://github.com/intel/llvm.git intel-llvm-git

cd $WORKDIR/intel-llvm-git

# configure llvm
python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=$WORKDIR/intel_llvm" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=$ROCM_PATH" --cmake-gen "Ninja"

cd $WORKDIR/intel-llvm-git/build
ninja all
ninja all lib/all tools/libdevice/libsycldevice
ninja install

cd $WORKDIR
```


### Testing the compiler
Just write this in an exemple file :
```c++ linenums="1" title="test.cpp"
#include <sycl/sycl.hpp>

int main(){

    size_t sz = 1000;

    sycl::buffer<int> buf (sz);

    std::cout << "device name : "
        << sycl::queue{}.get_device().get_info<sycl::info::device::name>()
        << std::endl;

    sycl::queue{}.submit([&](sycl::handler & cgh){

        sycl::accessor acc {buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
            acc[id] = id.get_linear_id();
        });

    }).wait();

    sycl::host_accessor acc {buf, sycl::read_only};

    std::cout << "expected : 999 | found : " << acc[sz-1] << std::endl;

}
```
compile it
```bash
export PATH=$HOMEDIR/.local/bin:$PATH
export LLVM_HOME=$SCRATCHDIR/intel_llvm/
echo "Intel LLVM dir  :" $LLVM_HOME
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH

clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=/opt/rocm-5.7.1 test.cpp
```

Allocate some time on the cluster to check if everything works, it should print the device name and 999, 8 times.
```bash
salloc -A cad14954 -N 1 -C "MI250" --job-name=interactive --time=100 --exclusive
srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest -- ./a.out
```




## Compiling Shamrock

Load the modules to compile Shamrock on adastra :

```bash
module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-intel
module load cray-mpich/8.1.26
module load cray-python
module load amd-mixed/5.7.1
module load rocm/5.7.1
```

Before running anything check if you have done the following commands.
If not the path to the compiler & python tools we have installed earlier will not be available

```bash
export PATH=$HOMEDIR/.local/bin:$PATH
export LLVM_HOME=$SCRATCHDIR/intel_llvm/
echo "Intel LLVM dir  :" $LLVM_HOME
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
```

```bash
cd Shamrock

cmake -S . -B build -G "Ninja" -DSYCL_IMPLEMENTATION=IntelLLVM -DCMAKE_CXX_COMPILER=$SCRATCHDIR/intel_llvm/bin/clang++ -DSHAMROCK_ENABLE_BACKEND=SYCL -DINTEL_LLVM_PATH=$SCRATCHDIR/intel_llvm -DCMAKE_C_COMPILER=$SCRATCHDIR/intel_llvm/bin/clang-19 -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=${ROCM_PATH} -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" -DBUILD_TEST=true -DCXX_FLAG_ARCH_NATIVE=off
```



# Old commands that were usefull for the hackaton

```

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-intel
module load cray-mpich/8.1.26
module load cray-python
module load amd-mixed/5.7.1

pip3 install -U cmake ninja
export PATH=/lus/home/CT10/cad14954/tdavidc/.local/bin:$PATH

module load rocm/5.7.1

python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=/lus/home/CT10/cad14954/tdavidc" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=$ROCM_PATH" --cmake-gen "Ninja"

ninja all\
    lib/all\
    tools/libdevice/libsycldevice\


export LLVM_HOME=$HOMEDIR/intel_llvm/
echo "Intel LLVM dir  :" $LLVM_HOME
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH


cmake -S . -B build -G "Ninja" -DSYCL_IMPLEMENTATION=IntelLLVM -DCMAKE_CXX_COMPILER=/lus/home/CT10/cad14954/tdavidc/intel_llvm/bin/clang++ -DSHAMROCK_ENABLE_BACKEND=SYCL -DINTEL_LLVM_PATH=/lus/home/CT10/cad14954/tdavidc/intel_llvm -DCMAKE_C_COMPILER=/lus/home/CT10/cad14954/tdavidc/intel_llvm/bin/clang-18 -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=/opt/rocm-5.7.1"

cd build

ninja

# to get mpich cray not the compiler
module load PrgEnv-intel

cmake -S . -B build -G "Ninja" -DSYCL_IMPLEMENTATION=IntelLLVM -DCMAKE_CXX_COMPILER=/lus/home/CT10/cad14954/tdavidc/intel_llvm/bin/clang++ -DSHAMROCK_ENABLE_BACKEND=SYCL -DINTEL_LLVM_PATH=/lus/home/CT10/cad14954/tdavidc/intel_llvm -DCMAKE_C_COMPILER=/lus/home/CT10/cad14954/tdavidc/intel_llvm/bin/clang-18 -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=${ROCM_PATH} -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" -DBUILD_TEST=true



```

salloc :
```
  607  salloc -A cad14954 -N 1 -C "MI250" --job-name=interactive --time=100 --exclusive
```


```
module purge
  252  module load cray-python
  253  pip -U ninja
  254  pip3 -U ninja
  255  pip3 install cmake ninja
  256  cd ..
  257  ls
  258  ls llvm/
  259  cd llvm/
  260  python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=$HOMEDIR/intel_llvm" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=$ROCM_PATH" --cmake-gen "Ninja"
  261  cat $HOMEDIR
  262  ninja
  263  pip3 install -U cmake ninja
  264  pip3 reinstall -U cmake ninja
  265  pip3 install -U --force-reinstall cmake ninja
  266  ninja
  267  cat ~/.bashrc
  268  export PATH=/lus/home/CT10/cad14954/tdavidc/.local/bin:$PATH
  269  ninja
  270  cmake --version
  271  python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=$HOMEDIR/intel_llvm" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=$ROCM_PATH" --cmake-gen "Ninja"
  272  rm -rf build
  273  module list
  274  module load rocm/5.7.1
  275  echo $ROCM_PATH
  276  python3 buildbot/configure.py --hip --cmake-opt="-DCMAKE_INSTALL_PREFIX=$HOMEDIR/intel_llvm" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=$ROCM_PATH" --cmake-gen "Ninja"
  277  htop
  278  cd build
  279  ninja
  280  ls
  281  ninja install
  282  history
  283  `ninja all\
    lib/all\
    tools/libdevice/libsycldevice\
  284  ninja all    lib/all    tools/libdevice/libsycldevice    install
  285  ninja all    lib/all    tools/libdevice/libsycldevice\
  286  ninja all lib/all tools/libdevice/libsycldevice
  287  ninja install
  288  hsitory
  289  history
```



```
cmake -S . -B build -G "Ninja" -DSYCL_IMPLEMENTATION=IntelLLVM -DCMAKE_CXX_COMPILER=~/intel_llvm/bin/clang++
```







MPI
```
-I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}
```

## Slurm scripts :

Slurm script exemple :

```bash linenums="1" title="slurm_script"
#!/bin/bash
#SBATCH --account=cad14954
#SBATCH --job-name=ShamrockScalling
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=%A.out
#SBATCH --time=0:01:00
#
echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo
#
cd $HOMEDIR/Shamrock/build
#
module purge
#
module load CCE-GPU-2.1.0
module load hipsycl
module load amd-mixed/5.5.1
module load cray-python
#

export MPICH_GPU_SUPPORT_ENABLED=1
export ACPP_DEBUG_LEVEL=2

srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest -- \
    ./shamrock --sycl-cfg 0:0 --loglevel 125 --smi \
    --rscript sedov_scale_test_updated.py
```



```

#!/bin/bash
#SBATCH --account=cad14954
#SBATCH --job-name=ShamrockScalling
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=%A.out
#SBATCH --time=0:01:00
#
echo "The job ${SLURM_JOB_ID} is running on these nodes:"
echo ${SLURM_NODELIST}
echo
#
cd $HOMEDIR/Shamrock/build
#
module purge
#
module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-intel
module load cray-mpich/8.1.26
module load cray-python
module load amd-mixed/5.7.1
#


export MPICH_GPU_SUPPORT_ENABLED=1
export ACPP_DEBUG_LEVEL=2

export LLVM_HOME=$HOMEDIR/intel_llvm
echo "Intel LLVM dir  :" $LLVM_HOME
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH

ldd ./shamrock

srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest -- \
    ./shamrock --sycl-cfg auto:HIP --loglevel 125 --smi-full \
    --rscript sedov_scale_test_updated.py
```










13816176.450450242
13707781.216734508

default no mpi
result rate : 16351506.630551208
result cnt : 16217760

default mpi aware
result rate : 16411686.111862464
result cnt : 16217760

bindings doc cines
result rate : 15812417.152442794
result cnt : 16217760

bindings ordered
result rate : 16758947.19113915
result cnt : 16217760

swap mystique
function Adastra_MI250_8TasksWith8ThreadsAnd1GPU() {
    # Node local rank 0 gets the GCD 0, is bound the cores [48-55] of NUMA domain 3 and uses the NIC 0
    # Node local rank 1 gets the GCD 1, is bound the cores [56-63] of NUMA domain 3 and uses the NIC 0
    # Node local rank 2 gets the GCD 2, is bound the cores [16-23] of NUMA domain 1 and uses the NIC 1
    # Node local rank 3 gets the GCD 3, is bound the cores [24-31] of NUMA domain 1 and uses the NIC 1
    # Node local rank 4 gets the GCD 4, is bound the cores [ 0- 7] of NUMA domain 0 and uses the NIC 2
    # Node local rank 5 gets the GCD 5, is bound the cores [ 8-15] of NUMA domain 0 and uses the NIC 2
    # Node local rank 6 gets the GCD 6, is bound the cores [32-39] of NUMA domain 2 and uses the NIC 3
    # Node local rank 7 gets the GCD 7, is bound the cores [40-47] of NUMA domain 2 and uses the NIC 3
    AFFINITY_NUMACTL=('48-55' '56-63' '16-23' '24-31' '0-7' '8-15' '32-39' '40-47')
    #AFFINITY_NUMACTL=('0-7' '8-15' '16-23' '24-31' '32-39' '40-47' '48-55' '56-63')
    AFFINITY_GPU=('0' '1' '2' '3' '4' '5' '6' '7')
    export MPICH_OFI_NIC_POLICY=GPU
}

function Adastra_MI250_8TasksWith8ThreadsAnd1GPU() {
    # Node local rank 0 gets the GCD 0, is bound the cores [48-55] of NUMA domain 3 and uses the NIC 0
    # Node local rank 1 gets the GCD 1, is bound the cores [56-63] of NUMA domain 3 and uses the NIC 0
    # Node local rank 2 gets the GCD 2, is bound the cores [16-23] of NUMA domain 1 and uses the NIC 1
    # Node local rank 3 gets the GCD 3, is bound the cores [24-31] of NUMA domain 1 and uses the NIC 1
    # Node local rank 4 gets the GCD 4, is bound the cores [ 0- 7] of NUMA domain 0 and uses the NIC 2
    # Node local rank 5 gets the GCD 5, is bound the cores [ 8-15] of NUMA domain 0 and uses the NIC 2
    # Node local rank 6 gets the GCD 6, is bound the cores [32-39] of NUMA domain 2 and uses the NIC 3
    # Node local rank 7 gets the GCD 7, is bound the cores [40-47] of NUMA domain 2 and uses the NIC 3
    AFFINITY_NUMACTL=('40-47' '56-63' '16-23' '24-31' '0-7' '8-15' '32-39' '48-55')
    #AFFINITY_NUMACTL=('0-7' '8-15' '16-23' '24-31' '32-39' '40-47' '48-55' '56-63')
    AFFINITY_GPU=('0' '1' '2' '3' '4' '5' '6' '7')
    export MPICH_OFI_NIC_POLICY=NUMA
}




result rate : 28686637.137355898
result cnt : 80474112


result rate : 28525822.08035385
result cnt : 80474112




result rate : 10474355.499388587
result cnt : 8157600

result rate : 9845472.799492419
result cnt : 8157600



result rate : 10992355.321209397
result cnt : 8157600
