# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

OMP_ROOT=`brew list libomp | grep libomp.a | sed -E "s/\/lib\/.*//"`

function setupcompiler {
    cmake -S $ACPP_GIT_DIR  -B $ACPP_BUILD_DIR \
        -DOpenMP_ROOT="${OMP_ROOT}" \
        -DWITH_SSCP_COMPILER=OFF \
        -DWITH_OPENCL_BACKEND=OFF \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR}
    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install)
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
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DBUILD_TEST=Yes
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC $MAKE_OPT)
}