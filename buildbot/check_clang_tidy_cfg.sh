#!/bin/bash

# set CLANG_TIDY_BIN to the argument if provided, otherwise use clang-tidy
CLANG_TIDY_BIN=${1:-"clang-tidy"}

echo "Using clang-tidy: $CLANG_TIDY_BIN"

# Get the path to the clang-tidy configuration file
CLANG_TIDY_CFG_PATH=".clang-tidy"

# Check if the file exists
if [ ! -f "$CLANG_TIDY_CFG_PATH" ]; then
    echo "Error: clang-tidy configuration file not found, run this script in the root of the repository"
    exit 1
fi

# generate a dummy file to test the clang-tidy configuration
echo "int main() { return 0; }" > test.cpp

# Check if the file is valid
$CLANG_TIDY_BIN --verify-config ./test.cpp
if [ $? -ne 0 ]; then
    echo "Error: clang-tidy configuration verification failed"
    rm -f test.cpp  # cleanup before exit
    exit 1
fi

# remove the dummy file
rm test.cpp
