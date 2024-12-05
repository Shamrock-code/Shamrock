export SHAMROCK_DIR=/home/tdavidcl/Documents/shamrock-dev/Shamrock
export BUILD_DIR=/home/tdavidcl/Documents/shamrock-dev/Shamrock/build

export CMAKE_GENERATOR="Ninja"

export MAKE_EXEC=ninja
export MAKE_OPT=()
export CMAKE_OPT=( -DSHAMROCK_USE_SHARED_LIB=On)
export SHAMROCK_BUILD_TYPE="Release"
export SHAMROCK_CXX_FLAGS=" --acpp-targets='omp'"

# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
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
