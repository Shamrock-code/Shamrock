# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

module use /soft/modulefiles
module load cmake
module load python
module load ninja

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH=$(dirname $(which icpx))/.. \
        -DCMAKE_CXX_COMPILER=$(which icpx) \
        -DCMAKE_C_COMPILER=$(which icx) \
        -DCMAKE_CXX_FLAGS="-fsycl -fp-model=precise" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
