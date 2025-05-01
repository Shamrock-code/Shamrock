# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

# test if python3-dev installed
# dpkg -l python3-dev
if ! dpkg -l python3.*-dev &> /dev/null; then
    echo "python3-dev is not installed. Installing it."
    apt update
    apt install -y python3-dev ninja-build python3-full python3-pip
fi

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH=$(dirname $(which icpx))/.. \
        -DCMAKE_CXX_COMPILER=$(which icpx) \
        -DCMAKE_CXX_FLAGS="-fsycl -fp-model=precise" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCXX_FLAG_ARCH_NATIVE=off \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
