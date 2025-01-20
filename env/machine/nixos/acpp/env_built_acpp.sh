# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

if [ -z ${ACPP_GIT_DIR+x} ]; then echo "ACPP_GIT_DIR is unset"; return; fi

if [ ! -f "$ACPP_GIT_DIR/README.md" ]; then
    echo " ------ Clonning AdaptiveCpp ------ "
    git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git $ACPP_GIT_DIR
    echo " ------  AdaptiveCpp Cloned  ------ "

fi

function setupcompiler {
    cmake \
        -S ${ACPP_GIT_DIR} \
        -B ${ACPP_BUILD_DIR} \
        -GNinja \
        -DCMAKE_INSTALL_PREFIX=$out \
        -DCLANG_INCLUDE_PATH=$CMAKE_CLANG_INCLUDE_PATH \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR}
    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install)
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ------ Compiling AdaptiveCpp ------ "
    setupcompiler
    echo " ------  AdaptiveCpp Compiled  ------ "

fi

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
