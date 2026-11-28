# Everything before this line will be provided by the new-env script

# Login nodes are shared, do not use all the cores
export MAKE_OPT=( -j 16)

module purge
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load boost/1.85.0--gcc--12.2.0
module load python/3.11.7
module load cmake/3.27.9
module load ninja/1.11.1

# The cuda module does not set CUDA_HOME
export CUDA_ROOT_DIR=$(dirname $(dirname $(which nvcc)))

# The system gcc is 8.5 (RHEL 8), the gcc module is needed for a C++20 libstdc++
export GCC_ROOT_DIR=$(dirname $(dirname $(which gcc)))
export GCC_LIB_DIR=$(dirname $(g++ -print-file-name=libstdc++.so))

export ACPP_TARGETS="cuda:sm_80" # Nvidia A100
export ACPP_VERSION=v25.10.0
export ACPP_APPDB_DIR=$BUILD_DIR/.env/acpp-appdb # otherwise it would we in the $HOME/.acpp

export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

export LLVM_VERSION=llvmorg-19.1.7
export LLVM_GIT_DIR=$BUILD_DIR/.env/llvm-git
export LLVM_BUILD_DIR=$BUILD_DIR/.env/llvm-build
export LLVM_INSTALL_DIR=$BUILD_DIR/.env/llvm-install

export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

function llvm_setup {

    echo " -> cleaning llvm build dirs ..."
    rm -rf "${LLVM_GIT_DIR:?}" "${LLVM_BUILD_DIR:?}"
    echo " -> done"

    clone_llvm || return

    cmake -S ${LLVM_GIT_DIR}/llvm -B ${LLVM_BUILD_DIR} \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR \
        -DCMAKE_INSTALL_RPATH=$LLVM_INSTALL_DIR/lib \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;openmp" \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
        -DLLVM_INCLUDE_BENCHMARKS=0 \
        -DLLVM_ENABLE_OCAMLDOC=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
        -DLLVM_ENABLE_DUMP=OFF  || return

    (cd ${LLVM_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install)  || return

    # Make clang use the gcc module toolchain instead of the system gcc 8.5
    for drv in clang clang++; do
        echo "--gcc-toolchain=${GCC_ROOT_DIR}" > ${LLVM_INSTALL_DIR}/bin/${drv}.cfg
    done

    local testdir
    testdir=$(mktemp -d "${LLVM_BUILD_DIR}/sanity-check.XXXXXX") || return
    (
        cd "$testdir" &&
        echo "int main() { return 0; }" > test.cpp &&
        "${LLVM_INSTALL_DIR}/bin/clang++" -O3 -fopenmp test.cpp &&
        ./a.out
    )
    local status=$?
    rm -rf "${testdir:?}"
    return $status

}

if [ ! -f "$LLVM_INSTALL_DIR/bin/clang++" ]; then
    echo " ----- llvm is not configured, compiling it ... -----"
    llvm_setup || return
    echo " ----- llvm configured ! -----"
fi

function setupcompiler {
    clone_acpp || return
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
        -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm/ \
        -DCUDAToolkit_ROOT=${CUDA_ROOT_DIR} \
        -DWITH_CUDA_BACKEND=ON \
        -DWITH_ROCM_BACKEND=Off \
        -DWITH_LEVEL_ZERO_BACKEND=Off \
        -DWITH_OPENCL_BACKEND=Off || return

    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- acpp configured ! -----"
fi

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,${GCC_LIB_DIR}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,${GCC_LIB_DIR}" \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
