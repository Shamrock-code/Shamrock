# Everything before this line will be provided by the new-env script

if which ccache &> /dev/null; then
    export CCACHE_CMAKE_ARG="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    echo " ----- ccache found, using it ----- "
else
    export CCACHE_CMAKE_ARG=""
fi

export INTELLLVM_INSTALL_DIR=/home/docker/compilers/DPCPP
export LD_LIBRARY_PATH=$INTELLLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        ${CCACHE_CMAKE_ARG} \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTELLLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTELLLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 --rocm-path=/opt/rocm" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
