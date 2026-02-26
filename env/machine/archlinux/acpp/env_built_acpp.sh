# Everything before this line will be provided by the new-env script

if which ccache &> /dev/null; then
    # to debug
    #export CCACHE_DEBUG=1
    #export CCACHE_DEBUGDIR=$BUILD_DIR/ccache-debug

    export CCACHE_COMPILERTYPE=clang
    export CCACHE_CMAKE_ARG="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    echo " ----- ccache found, using it ----- "
else
    export CCACHE_CMAKE_ARG=""
fi

# Arch linux check required packages
echo " ---------- Activating sham environment ---------- "
echo "Checking required packages..."

missing_packages=()
all_packages=(
    "base-devel"
    "git"
    "python"
    "cmake"
    "boost"
    "ninja"
    "openmp"
    "openmpi"
    "doxygen"
    "llvm19"
    "clang19"
    "lld"
)

for package in "${all_packages[@]}"; do
    if pacman -Q "$package" >/dev/null 2>&1; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is NOT installed"
        missing_packages+=("$package")
    fi
done

echo ""
if [ ${#missing_packages[@]} -eq 0 ]; then
    echo "All required packages are installed!"
else
    echo "Missing packages: ${missing_packages[*]}"
    echo "Install all missing packages with: sudo pacman -S ${missing_packages[*]}"
fi
echo " ------------- Environment activated ------------- "



export ACPP_VERSION=develop
export ACPP_APPDB_DIR=/tmp/acpp-appdb # otherwise it would we in the $HOME/.acpp
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir
export LLVM_INSTALL_DIR=/usr/lib/llvm19

function setupcompiler {
    clone_acpp || return
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        ${CCACHE_CMAKE_ARG} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
        -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm/ \
        -DACPP_LLD_PATH=/usr/bin/ld.lld \
        || return
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
        ${CCACHE_CMAKE_ARG} \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
