#!/bin/bash
if [ -z ${INTEL_LLVM_GIT_DIR+x} ]; then echo "INTEL_LLVM_GIT_DIR is unset"; return; fi

if [ ! -f "$INTEL_LLVM_GIT_DIR/README.md" ]; then
    echo " ------ Clonning LLVM ------ "

    if [ -z ${INTEL_LLVM_VERSION+x} ]
    then
        echo "-> git clone --depth 1 -b sycl https://github.com/intel/llvm.git $INTEL_LLVM_GIT_DIR"
        git clone --depth 1 -b sycl https://github.com/intel/llvm.git $INTEL_LLVM_GIT_DIR
    else
        echo "-> git clone --depth 1 -b $INTEL_LLVM_VERSION https://github.com/intel/llvm.git $INTEL_LLVM_GIT_DIR"
        git clone --depth 1 -b $INTEL_LLVM_VERSION https://github.com/intel/llvm.git $INTEL_LLVM_GIT_DIR
    fi
    echo " ------  LLVM Cloned  ------ "

fi
