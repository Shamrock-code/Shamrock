# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

export LD_LIBRARY_PATH=$INTELLLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

setupcompiler () {

    python3 ${INTELLLVM_GIT_DIR}/buildbot/configure.py \
        "${INTELLLVM_CONFIGURE_ARGS[@]}" \
        --cmake-gen "${CMAKE_GENERATOR}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTELLLVM_INSTALL_DIR}"

    (cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice)
    (cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC install)
    # ninja
    #(cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all tools/libdevice/libsycldevice)
    # make
    #(cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice)
}

updatecompiler () {
    (cd ${ACPP_GIT_DIR} && git pull)
    setupcompiler
}

shamconfigure () {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTELLLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTELLLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes
}

shammake () {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}