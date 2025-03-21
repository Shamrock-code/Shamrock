#!/bin/bash
if [ -z ${LLVM_GIT_DIR+x} ]; then echo "LLVM_GIT_DIR is unset"; return; fi

if [ ! -f "$LLVM_GIT_DIR/README.md" ]; then
    echo " ------ Clonning LLVM ------ "

    if [ -z ${LLVM_VERSION+x} ]
    then
        echo "-> git clone --depth 1 https://github.com/llvm/llvm-project.git $LLVM_GIT_DIR"
        git clone --depth 1 https://github.com/llvm/llvm-project.git $LLVM_GIT_DIR
    else
        echo "-> git clone --depth 1 -b $LLVM_VERSION https://github.com/llvm/llvm-project.git $LLVM_GIT_DIR"
        git clone --depth 1 -b $LLVM_VERSION https://github.com/llvm/llvm-project.git $LLVM_GIT_DIR
    fi

    echo " ------  LLVM Cloned  ------ "

fi
