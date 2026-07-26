# Everything before this line will be provided by the new-env script

module use /soft/modulefiles
module load cmake
module load python
module load ninja

function shamconfigure {
    # note that the -g flag is set. In principle there is no impact on the perf, but if you run on
    # aurora it is better to enable debug symbols for crash reporting.
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH=$(dirname $(which icpx))/.. \
        -DCMAKE_CXX_COMPILER=$(which icpx) \
        -DCMAKE_C_COMPILER=$(which icx) \
        -DCMAKE_CXX_FLAGS="-g -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \"-device pvc\" -fp-model=precise -fno-system-debug --offload-compress " \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries -flink-huge-device-code" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        -DSHAMROCK_USE_CPPTRACE=Yes \
        -DSHAMROCK_USE_GEOPM=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
