# Everything before this line will be provided by the new-env script
module load env/release/2025.1
module load foss/2025a
module load AdaptiveCpp/25.10.0-GCC-14.2.0-CUDA-12.8.0
module load CMake/3.31.3-GCCcore-14.2.0
module load Ninja/1.12.1-GCCcore-14.2.0
module load Python/3.13.1-GCCcore-14.2.0

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_C_COMPILER=${EBROOTLLVM}/bin/clang \
        -DCMAKE_CXX_COMPILER=${EBROOTADAPTIVECPP}/bin/acpp \
        -DACPP_PATH=${EBROOTADAPTIVECPP} \
        -DCMAKE_CXX_FLAGS=" --acpp-targets='omp;cuda:sm_80'" \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
