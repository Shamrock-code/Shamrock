# Everything before this line will be provided by the new-env script

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

export REF_FILES_PATH=$BUILD_DIR/reference-files

function pull_reffiles {
    git clone https://github.com/Shamrock-code/reference-files.git $REF_FILES_PATH
}

function update_env {
    conda env update --file environment.yml --prune
}
