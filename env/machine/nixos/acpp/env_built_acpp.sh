# Everything before this line will be provided by the new-env script

export ACPP_VERSION=v24.10.0
export ACPP_APPDB_DIR=/tmp/acpp-appdb # otherwise it would we in the $HOME/.acpp
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

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
