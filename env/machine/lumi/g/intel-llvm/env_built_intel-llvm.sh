# Everything before this line will be provided by the new-env script

# On LUMI using the default (256) result in killed jobs as the login node is destroyed ^^
export MAKE_OPT=( -j 128)
export NINJA_STATUS="[%f/%t j=%r] "

module purge

module load LUMI/24.03
module load partition/G
module load cray-python
module load rocm/6.0.3
module load cpeAMD/24.03 # For MPIch

export MPICH_GPU_SUPPORT_ENABLED=1

export PATH=$HOME/.local/bin:$PATH
pip3 install -U ninja cmake

export INTEL_LLVM_VERSION=v6.0.0
export INTEL_LLVM_GIT_DIR=/tmp/intelllvm-git
export INTEL_LLVM_INSTALL_DIR=$BUILD_DIR/.env/intelllvm-install

export PATH=$INTEL_LLVM_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$INTEL_LLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

function setupcompiler {

    echo " -> cleaning intel/llvm build dirs ..."
    rm -rf ${INTEL_LLVM_GIT_DIR} || return
    echo " -> done"

    clone_intel_llvm || return

    echo " ---- Running compiler setup ----"

    # See : https://dci.dci-gitlab.cines.fr/webextranet/software_stack/libraries/index.html#compiling-intel-llvm
    #cd ${INTEL_LLVM_GIT_DIR}

    python3 ${INTEL_LLVM_GIT_DIR}/buildbot/configure.py \
        --hip \
        --cmake-opt="-DCMAKE_C_COMPILER=amdclang" \
        --cmake-opt="-DCMAKE_CXX_COMPILER=amdclang++" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTEL_LLVM_INSTALL_DIR}" \
        --cmake-opt="-DUR_HIP_ROCM_DIR=${ROCM_PATH}" \
        --cmake-gen="Ninja" || return

    (cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return

}

if [ ! -f "${INTEL_LLVM_INSTALL_DIR}/bin/clang++" ]; then
    echo " ----- intel llvm is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- intel llvm configured ! -----"
fi

function shamconfigure {

    # see https://github.com/audacity/audacity/issues/3173 for why "-Wl,--copy-dt-needed-entries"
    # is needed
    # If not one gets /usr/lib64/gcc/x86_64-suse-linux/13/../../../../x86_64-suse-linux/bin/ld: /lib64/libpthread.so.0: error adding symbols: DSO missing from command line

    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTEL_LLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTEL_LLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_C_COMPILER="${INTEL_LLVM_INSTALL_DIR}/bin/clang" \
        -DCMAKE_CXX_FLAGS="-march=znver3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=${ROCM_PATH} -isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries -L"${CRAY_MPICH_PREFIX}/lib" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
