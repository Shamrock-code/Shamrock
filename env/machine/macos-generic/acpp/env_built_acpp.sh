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

# List of required packages
required_packages=("cmake" "libomp" "boost" "open-mpi" "adaptivecpp")

echo " ---------- Activating sham environment ---------- "
# Check if each package is installed
for package in "${required_packages[@]}"; do
    if ! brew list --versions "$package" &>/dev/null; then
        echo "Error: $package is not installed. Please run 'brew install $package'."
        return 1  # Abort sourcing the script and return to the current shell
    else
        echo "$package is installed."
    fi
done

echo "All required packages are installed."

ACPP_ROOT=`brew list adaptivecpp | grep acpp-info | sed -E "s/\/bin\/.*//"`
echo " ------------- Environment activated ------------- "

export CCACHE_DEBUG=1
export CCACHE_DEBUGDIR=$BUILD_DIR/ccache-debug

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        ${CCACHE_CMAKE_ARG} \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="acpp" \
        -DCMAKE_CXX_FLAGS="-I$OMP_ROOT/include" \
        -DACPP_PATH="${ACPP_ROOT}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
