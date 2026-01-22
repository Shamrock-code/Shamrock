# Everything before this line will be provided by the new-env script

export MAKE_OPT=( -j 128)
export NINJA_STATUS="[%f/%t j=%r] "

module --force purge

module load LUMI/24.03
module load partition/G
module load cray-python
module load rocm/6.0.3
module load Boost/1.83.0-cpeAMD-24.03

export MPICH_GPU_SUPPORT_ENABLED=1

export PATH=$HOME/.local/bin:$PATH
pip3 install -U ninja cmake

export ACPP_VERSION=v24.10.0
export ACPP_TARGETS="hip:gfx90a"
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

export C_INCLUDE_PATH=$ROCM_PATH/llvm/include
export CPLUS_INCLUDE_PATH=$ROCM_PATH/llvm/include

#export LUMI_WORKSPACE_OUTPUT=$(lumi-workspaces)
#export PROJECT_SCRATCH=$(echo $LUMI_WORKSPACE_OUTPUT | grep -o '/scratch[^ ]*')
#export PROJECT_NUM=$(echo $LUMI_WORKSPACE_OUTPUT | grep -o '/scratch/[^ ]*' | cut -d'/' -f3)

function setupcompiler {

    clone_acpp || return

    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DROCM_PATH=$ROCM_PATH \
        -DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
        -DWITH_ACCELERATED_CPU=ON \
        -DWITH_CPU_BACKEND=ON \
        -DWITH_CUDA_BACKEND=OFF \
        -DWITH_ROCM_BACKEND=ON \
        -DWITH_OPENCL_BACKEND=OFF \
        -DWITH_LEVEL_ZERO_BACKEND=OFF \
        -DACPP_TARGETS="gfx90a" \
        -DBoost_NO_BOOST_CMAKE=TRUE \
        -DBoost_NO_SYSTEM_PATHS=TRUE \
        -DWITH_SSCP_COMPILER=OFF \
        -DLLVM_DIR=${ROCM_PATH}/llvm/lib/cmake/llvm/  || return

    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- acpp configured ! -----"
fi


function shamconfigure {
    # Why the FFFF is pthread not linked by default ?
    # If one invoke a c++ thread that compile to something that uses pthread

    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCMAKE_CXX_FLAGS="-march=znver3 -isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-lpthread -L"${CRAY_MPICH_PREFIX}/lib" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
