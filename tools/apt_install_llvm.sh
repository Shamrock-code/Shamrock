#!/bin/bash

set -e

# Retry up to 10 times only while the command fails and its output contains 404.
retry_on_404() {
    local out rc i

    # Loop until success or 10 attempts
    for ((i = 1; i <= 10; i++)); do

        # Run the command and capture output
        out=$(mktemp)
        "$@" 2>&1 | tee "$out"

        # Capture return code
        rc=${PIPESTATUS[0]}

        # If command succeeded, remove temporary file and return
        if [[ $rc -eq 0 ]]; then
            rm -f "$out"
            return 0
        fi

        # If command failed and output contains 404 or "Network is unreachable", continue
        if grep -q -e 404 -e "Network is unreachable" "$out"; then
            rm -f "$out"
            sleep 2
            continue
        fi

        # If command failed and output does not contain 404, return
        rm -f "$out"
        return "$rc"
    done
    return 1
}

if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <llvm_version>"
    exit 1
fi

LLVM_VERSION=$1
echo "-> Installing LLVM: $LLVM_VERSION"

echo "-> Downloading LLVM script"
retry_on_404 wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh

echo "-> sudo ./llvm.sh $LLVM_VERSION"
retry_on_404 sudo ./llvm.sh "$LLVM_VERSION"

echo "-> sudo apt install -y <llvm stuff>"
retry_on_404 sudo apt install -y "libclang-${LLVM_VERSION}-dev" "clang-tools-${LLVM_VERSION}" "libomp-${LLVM_VERSION}-dev"

echo "-> Updating symlink for clang"
if [[ "$LLVM_VERSION" == "16" ]]; then
    sudo rm -r /usr/lib/clang/$LLVM_VERSION*
    sudo ln -s /usr/lib/llvm-$LLVM_VERSION/lib/clang/$LLVM_VERSION /usr/lib/clang/$LLVM_VERSION
fi

rm llvm.sh
