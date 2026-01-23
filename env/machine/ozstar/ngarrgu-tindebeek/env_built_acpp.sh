# Everything before this line will be provided by the new-env script
export ACPP_VERSION=v25.02.0
export ACPP_APPDB_DIR=/tmp/acpp-appdb # otherwise it would be in the $HOME/.acpp
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir
export SHAMROCK_CXX_FLAGS=" --acpp-targets='omp;cuda:sm_80'"

export NINJA_STATUS="[%f/%t j=%r] "

module load \
    gcc/13.2.0 \
    openmpi/4.1.6 \
    llvm/20.1.5 \
    cmake/3.27.6 \
    boost.mpi/1.83.0 \
    cuda/12.8.0 \
    ninja/1.11.1 \

function setupcompiler {
    clone_acpp || return
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} || return
    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- acpp configured ! -----"
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
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
