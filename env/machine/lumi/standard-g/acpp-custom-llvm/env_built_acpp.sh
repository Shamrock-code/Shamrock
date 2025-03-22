#### Before this lines are env specific definitions

# On LUMI using the default (256) result in killed jobs as the login node is destroyed ^^
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

export LLVM_VERSION=llvmorg-18.1.8
export LLVM_GIT_DIR=/tmp/llvm-git
export LLVM_BUILD_DIR=/tmp/llvm-build
export LLVM_INSTALL_DIR=$BUILD_DIR/.env/llvm-install

export ACPP_VERSION=v24.10.0
export ACPP_APPDB_DIR=/tmp/acpp-appdb # otherwise it would we in the $HOME/.acpp

case "$ACPPMODE" in
    "SSCP")
        export ACPP_TARGETS=hip:gfx90a
        ;;
    "SMCP")
        export ACPP_TARGETS=generic
        ;;
    *)
        echo "Unknown ACPPMODE: $ACPPMODE"
        return
        ;;
esac

. $BUILD_DIR/.env/clone-acpp

export C_INCLUDE_PATH=$ROCM_PATH/llvm/include
export CPLUS_INCLUDE_PATH=$ROCM_PATH/llvm/include

#export LUMI_WORKSPACE_OUTPUT=$(lumi-workspaces)
#export PROJECT_SCRATCH=$(echo $LUMI_WORKSPACE_OUTPUT | grep -o '/scratch[^ ]*')
#export PROJECT_NUM=$(echo $LUMI_WORKSPACE_OUTPUT | grep -o '/scratch/[^ ]*' | cut -d'/' -f3)

function llvm_setup {
    set -e

    echo " -> cleaning llvm build dirs ..."
    rm -rf ${LLVM_GIT_DIR} ${LLVM_BUILD_DIR}
    echo " -> done"

    . $BUILD_DIR/.env/clone-llvm

    cmake -S ${LLVM_GIT_DIR}/llvm -B ${LLVM_BUILD_DIR} \
        -DCMAKE_C_COMPILER=gcc-13 \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR \
        -DCMAKE_INSTALL_RPATH=$LLVM_INSTALL_DIR/lib \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;openmp" \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
        -DLLVM_INCLUDE_BENCHMARKS=0 \
        -DLLVM_ENABLE_OCAMLDOC=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
        -DLLVM_ENABLE_DUMP=OFF

    (cd ${LLVM_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install)

    echo "int main() { return 0; }" > test.cpp
    ${LLVM_INSTALL_DIR}/bin/clang++ -O3 -fopenmp test.cpp
    ./a.out
    rm a.out test.cpp

    set +e
}

if [ ! -f "$LLVM_INSTALL_DIR/bin/clang++" ]; then
    echo " ----- llvm is not configured, compiling it ... -----"
    llvm_setup
    echo " ----- llvm configured ! -----"
fi

function setupcompiler {
    set -e
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DROCM_PATH=$ROCM_PATH \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
        -DBoost_NO_BOOST_CMAKE=TRUE \
        -DBoost_NO_SYSTEM_PATHS=TRUE \
        -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm/

    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install)
    set +e
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler
    echo " ----- acpp configured ! -----"
fi

function shamconfigure {

    # see https://github.com/audacity/audacity/issues/3173 for why "-Wl,--copy-dt-needed-entries" 
    # is needed 
    # If not one gets /usr/lib64/gcc/x86_64-suse-linux/13/../../../../x86_64-suse-linux/bin/ld: /lib64/libpthread.so.0: error adding symbols: DSO missing from command line

    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCMAKE_CXX_FLAGS="-march=znver3 -isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries -L"${CRAY_MPICH_PREFIX}/lib" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
