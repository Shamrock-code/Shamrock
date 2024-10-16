#!/bin/bash
if [ -z ${INTELLLVM_GIT_DIR+x} ]; then echo "INTELLLVM_GIT_DIR is unset"; return; fi

if [ ! -f "$INTELLLVM_GIT_DIR/README.md" ]; then
    echo " ------ Clonning LLVM ------ "

    if [ -z ${INTEL_LLVM_VERSION+x} ]
    then
        echo "-> git clone -b sycl https://github.com/intel/llvm.git $INTELLLVM_GIT_DIR"
        git clone -b sycl https://github.com/intel/llvm.git $INTELLLVM_GIT_DIR
        (cd $INTELLLVM_GIT_DIR && git checkout $INTEL_LLVM_VERSION)
    else
        echo "-> git clone --depth 1 -b sycl https://github.com/intel/llvm.git $INTELLLVM_GIT_DIR"
        git clone --depth 1 -b sycl https://github.com/intel/llvm.git $INTELLLVM_GIT_DIR
    fi
    echo " ------  LLVM Cloned  ------ "

fi
