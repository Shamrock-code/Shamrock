
# Everything before this line will be provided by the new-env script
ACPP_TARGETS="omp"
WITH_CUDA_BACKEND=Off
WITH_ROCM_BACKEND=Off
WITH_SSCP_COMPILER=Off

CBP_CUDA_DIR=/usr/lib/cuda
CBP_ROCM_DIR=/opt/rocm

if [ "${ACPP_BACKEND}" = "cuda" ]; then
    GPU_TARGET=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "." | sort -n | tail -1)
    ACPP_TARGETS+=";cuda:sm_${GPU_TARGET}"
    LLVM_TARGET="NVPTX"
    GPU_LIBRARY_FLAG="-DCUDAToolkit_LIBRARY_ROOT=${CBP_CUDA_DIR}"
    WITH_CUDA_BACKEND=On

elif [ "${ACPP_BACKEND}" = "rocm" ]; then
    GPU_TARGET=$(rocminfo | \grep --color=never -oE 'gfx[0-9]+[a-zA-Z0-9]*' | uniq)
    ACPP_TARGETS+=";hip:${GPU_TARGET}"
    LLVM_TARGET="AMDGPU"
    GPU_LIBRARY_FLAG="-DROCM_PATH=${CBP_ROCM_DIR}"
    WITH_ROCM_BACKEND=On

elif [ "${ACPP_BACKEND}" = "sscp" ]; then
    ACPP_TARGETS="generic"
    WITH_SSCP_COMPILER=On

    if nvidia-smi &> /dev/null; then
        LLVM_TARGET="NVPTX"
        GPU_LIBRARY_FLAG="-DCUDAToolkit_LIBRARY_ROOT=${CBP_CUDA_DIR}"
        WITH_CUDA_BACKEND=On

    elif rocm-smi &> /dev/null; then
        LLVM_TARGET="AMDGPU"
        GPU_LIBRARY_FLAG="-DROCM_PATH=${CBP_ROCM_DIR}"
        WITH_ROCM_BACKEND=On
    fi
fi

export ACPP_VERSION=develop
export ACPP_APPDB_DIR=$BUILD_DIR/.env/acpp-appdb # otherwise it would we in the $HOME/.acpp

export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

export ACPP_DEBUG_LEVEL=0

export LLVM_VERSION=llvmorg-19.1.7
export LLVM_GIT_DIR=$BUILD_DIR/.env/llvm-git
export LLVM_BUILD_DIR=$BUILD_DIR/.env/llvm-build
export LLVM_INSTALL_DIR=$BUILD_DIR/.env/llvm-install

export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

function llvm_setup {

    echo " -> cleaning llvm build dirs ..."
    rm -rf ${LLVM_GIT_DIR} ${LLVM_BUILD_DIR}
    echo " -> done"

    clone_llvm || return

    cmake -S ${LLVM_GIT_DIR}/llvm -B ${LLVM_BUILD_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR \
        -DCMAKE_INSTALL_RPATH=$LLVM_INSTALL_DIR/lib \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;openmp" \
        -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGET};X86" \
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

    echo "int main() { return 0; }" > test.cpp
    ${LLVM_INSTALL_DIR}/bin/clang++ -O3 -fopenmp test.cpp  || return
    ./a.out  || return
    rm a.out test.cpp

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
        ${GPU_LIBRARY_FLAG} \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
        -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm/ \
        -DWITH_CUDA_BACKEND=${WITH_CUDA_BACKEND} \
        -DWITH_ROCM_BACKEND=${WITH_ROCM_BACKEND} \
        -DWITH_LEVEL_ZERO_BACKEND=Off \
        -DWITH_SSCP_COMPILER=${WITH_SSCP_COMPILER} || return

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
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCMAKE_CXX_FLAGS="--acpp-targets=\"${ACPP_TARGETS}\"" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
