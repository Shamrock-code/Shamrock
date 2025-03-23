# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

export INTELLLVM_INSTALL_DIR=/home/docker/compilers/DPCPP
export LD_LIBRARY_PATH=$INTELLLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
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

export REF_FILES_PATH=$BUILD_DIR/reference-files

function pull_reffiles {
    git clone https://github.com/Shamrock-code/reference-files.git $REF_FILES_PATH
}
