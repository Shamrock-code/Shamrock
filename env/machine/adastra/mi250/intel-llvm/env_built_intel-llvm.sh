# Everything before this line will be provided by the new-env script

export INTEL_LLVM_VERSION=v6.0.0
export INTEL_LLVM_GIT_DIR=/tmp/intelllvm-git
export INTEL_LLVM_INSTALL_DIR=$BUILD_DIR/.env/intelllvm-install

function loadmodules {
    module purge

    module load cpe/23.12
    module load craype-accel-amd-gfx90a craype-x86-trento
    module load PrgEnv-intel
    module load cray-mpich/8.1.26
    module load cray-python
    module load amd-mixed/5.7.1
    module load rocm/5.7.1
    module load cmake/3.27.9
}

loadmodules

export PATH=$HOMEDIR/.local/bin:$PATH

export PATH=$INTEL_LLVM_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$INTEL_LLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

export MPICH_GPU_SUPPORT_ENABLED=1

function setupcompiler {
    echo " ---- Running compiler setup ----"

    echo " -- Restoring env default"
    source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
    echo " -- module purge"
    module purge
    source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
    module list

    # See : https://dci.dci-gitlab.cines.fr/webextranet/software_stack/libraries/index.html#compiling-intel-llvm
    echo " -- clone intel/llvm"
    clone_intel_llvm || return
    cd ${INTEL_LLVM_GIT_DIR}

    module purge

    module load cpe/23.12
    module load cray-python
    module load rocm/5.7.1
    module load cmake/3.27.9

    module list

    pip3 install -U ninja

    python3 buildbot/configure.py \
        --hip \
        --cmake-opt="-DCMAKE_C_COMPILER=amdclang" \
        --cmake-opt="-DCMAKE_CXX_COMPILER=amdclang++" \
        --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=${ROCM_PATH}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTEL_LLVM_INSTALL_DIR}" \
        --cmake-gen="Ninja"

    cd build

    time ninja -k0 all lib/all tools/libdevice/libsycldevice
    time ninja -k0 install

    module purge
    loadmodules

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
        -DCMAKE_C_COMPILER="${INTEL_LLVM_INSTALL_DIR}/bin/clang" \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=${ROCM_PATH} -isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-L"${CRAY_MPICH_PREFIX}/lib" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lpthread" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
