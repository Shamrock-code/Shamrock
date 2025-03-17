# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

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

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
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

export REF_FILES_PATH=$BUILD_DIR/reference-files

function pull_reffiles {
    git clone https://github.com/Shamrock-code/reference-files.git $REF_FILES_PATH
}
