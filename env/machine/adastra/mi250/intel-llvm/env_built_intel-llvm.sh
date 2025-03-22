# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR
function loadmodules {
    module purge

    module load cpe/23.12
    module load craype-accel-amd-gfx90a craype-x86-trento
    module load PrgEnv-intel
    module load cray-mpich/8.1.26
    module load cray-python
    module load amd-mixed/5.7.1
    module load rocm/5.7.1
}

loadmodules

export PATH=$HOMEDIR/.local/bin:$PATH

export PATH=$INTELLLVM_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$INTELLLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

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
    echo " -- clone inte/llvm"
    git clone --branch="sycl" https://github.com/intel/llvm ${INTEL_LLVM_GIT_DIR} || true
    cd ${INTEL_LLVM_GIT_DIR}

    module purge

    module load cpe/23.12
    module load cray-python
    module load rocm/5.7.1

    module list

    pip3 install -U ninja cmake

    python3 buildbot/configure.py \
        --hip \
        --cmake-opt="-DCMAKE_C_COMPILER=amdclang" \
        --cmake-opt="-DCMAKE_CXX_COMPILER=amdclang++" \
        --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=${ROCM_PATH}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTELLLVM_INSTALL_DIR}" \
        --cmake-gen="Ninja"

    cd build

    time ninja -k0 all lib/all tools/libdevice/libsycldevice
    time ninja -k0 install

    module purge
    loadmodules

}

function updatecompiler {
    (cd ${ACPP_GIT_DIR} && git pull)
    setupcompiler
}

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTELLLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTELLLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_C_COMPILER="${INTELLLVM_INSTALL_DIR}/bin/clang" \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a --rocm-path=${ROCM_PATH} -isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-L"${CRAY_MPICH_PREFIX}/lib" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
