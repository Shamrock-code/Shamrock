# Everything before this line will be provided by the new-env script

module load intel/2024.0.1-sycl_cuda
module load impi/2021.11
module load mkl/2024.0
module load python/3.12.1
module load cmake/3.25.1
module load ninja/1.12.1
module load hdf5/1.14.1-2

export INTEL_ROOT=$INTEL_HOME

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH=$INTEL_ROOT \
        -DCMAKE_CXX_COMPILER=$INTEL_ROOT/bin/icpx \
        -DCMAKE_C_COMPILER=$INTEL_ROOT/bin/icx \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvidia_gpu_sm_80" \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}