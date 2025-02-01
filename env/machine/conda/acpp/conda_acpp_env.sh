# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

export CMAKE_GENERATOR="Ninja"

# Try to activate the conda environment
conda activate shamrock_dev_environment

# Check if the activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate 'shamrock_dev_environment'. Creating environment from environment.yml..."
    conda env create -f ./environment.yml
    conda activate shamrock_dev_environment
fi


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
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
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
