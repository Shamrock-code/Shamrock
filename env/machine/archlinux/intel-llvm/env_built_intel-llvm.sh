# Everything before this line will be provided by the new-env script




# Arch linux check required packages
echo " ---------- Activating sham environment ---------- "
echo "Checking required packages..."

local missing_packages=()
local all_packages=("python" "cmake" "clang" "llvm" "boost" "ninja" "openmp" "openmpi" "doxygen")

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




export INTEL_LLVM_VERSION=v6.0.0
export INTEL_LLVM_GIT_DIR=/tmp/intelllvm-git
export INTEL_LLVM_INSTALL_DIR=$BUILD_DIR/.env/intelllvm-install
clone_intel_llvm || return

export LD_LIBRARY_PATH=$INTEL_LLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

function setupcompiler {

    python3 ${INTEL_LLVM_GIT_DIR}/buildbot/configure.py \
        "${INTEL_LLVM_CONFIGURE_ARGS[@]}" \
        --cmake-gen "${CMAKE_GENERATOR}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTEL_LLVM_INSTALL_DIR}" || return

    (cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice) || return
    (cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC install) || return
    # ninja
    #(cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all tools/libdevice/libsycldevice)
    # make
    #(cd ${INTEL_LLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice)
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
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
