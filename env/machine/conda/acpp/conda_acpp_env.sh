# Everything before this line will be provided by the new-env script

if which ccache &> /dev/null; then
    # to debug
    #export CCACHE_DEBUG=1
    #export CCACHE_DEBUGDIR=$BUILD_DIR/ccache-debug

    export CCACHE_COMPILERTYPE=clang
    export CCACHE_CMAKE_ARG="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    echo " ----- ccache found, using it ----- "
else
    export CCACHE_CMAKE_ARG=""
fi

export CMAKE_GENERATOR="Ninja"
export ACPP_APPDB_DIR=/tmp/acpp-appdb # otherwise it would we in the $HOME/.acpp

# Check if the activation failed
if ! conda info --envs | grep -q "shamrock_dev_environment"; then
    echo " -- Shamrock dev environment not found."
    echo " --------- Creating environment from environment.yml... --------- "
    conda env create -f ./environment.yml
    echo " --------------------- Environment created ---------------------- "
fi

# Try to activate the conda environment
echo " --------- Activating conda environment --------- "
conda activate shamrock_dev_environment
echo " ------------- Environment activated ------------ "

#enfore the use of the correct clang++ as the installation of acpp register a buggy compiler
export ACPP_CPU_CXX=$(which clang++)

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        ${CCACHE_CMAKE_ARG} \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="acpp" \
        -DCMAKE_C_COMPILER="clang" \
        -DCMAKE_MAKE_PROGRAM=$(which ninja) \
        -DUSE_SYSTEM_FMTLIB=On \
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS} --acpp-cpu-cxx=$(which clang++)" \
        -DACPP_PATH=$(which acpp) \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}

function update_env {
    conda env update --file environment.yml --prune
}
