
# Everything before this line will be provided by the new-env script

SYCL_TARGET="spir64_x86_64"
if nvidia-smi &> /dev/null; then
    GPU_LLVM_FLAG=(--cuda --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda")
    GPU_TARGET=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "." | sort -n | tail -1)
    SYCL_TARGET+=",nvidia_gpu_sm_${GPU_TARGET}"

elif rocm-smi &> /dev/null; then
    GPU_LLVM_FLAG=(--hip --cmake-opt="-DUR_HIP_ROCM_DIR=/opt/rocm")
    GPU_TARGET=$(rocminfo | \grep --color=never -oE "gfx[0-9]+[a-zA-Z0-9]*" | uniq)
    SYCL_TARGET="amd_gpu_${GPU_TARGET} -Wno-deprecated-pragma"
fi

export INTEL_LLVM_VERSION=v6.2.0
export INTEL_LLVM_GIT_DIR=$BUILD_DIR/.env/intelllvm-git
export INTEL_LLVM_INSTALL_DIR=$BUILD_DIR/.env/intelllvm-install

export LD_LIBRARY_PATH=$INTEL_LLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

function setupcompiler {

    clone_intel_llvm || return

    python3 ${INTEL_LLVM_GIT_DIR}/buildbot/configure.py \
        "${GPU_LLVM_FLAG[@]}" \
        --cmake-gen "${CMAKE_GENERATOR}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTEL_LLVM_INSTALL_DIR}" || return

    (cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice) || return
    (cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC install) || return

}

if [ ! -f "${INTEL_LLVM_INSTALL_DIR}/bin/clang++" ]; then
    echo " ----- intel llvm is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- intel llvm configured ! -----"
fi


function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTEL_LLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTEL_LLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=${SYCL_TARGET}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
