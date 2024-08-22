# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

export LD_LIBRARY_PATH=$INTELLLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

function setupcompiler {

    python3 ${INTELLLVM_GIT_DIR}/buildbot/configure.py \
        --cuda \
        --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda" \
        --cmake-gen "${CMAKE_GENERATOR}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTELLLVM_INSTALL_DIR}"

    (cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice)
    (cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC install)

}

function updatecompiler {
    (cd ${ACPP_GIT_DIR} && git pull)
    setupcompiler
}

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTELLLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTELLLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvidia_gpu_sm_80" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}

export REF_FILES_PATH=$BUILD_DIR/reference-files

function pull_reffiles {
    git clone git@github.com:Shamrock-code/reference-files.git $REF_FILES_PATH
}
