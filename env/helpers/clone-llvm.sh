#!/bin/bash
function clone_llvm {

    LLVM_URL="https://github.com/llvm/llvm-project.git"

    if [ -z ${LLVM_GIT_DIR+x} ]; then echo "LLVM_GIT_DIR is unset"; return 1; fi

    if [ ! -f "$LLVM_GIT_DIR/README.md" ]; then
        echo " ------ Clonning LLVM ------ "

        if [ -z ${LLVM_VERSION+x} ]
        then
            echo "-> git clone --depth 1 $LLVM_URL $LLVM_GIT_DIR"
            git clone --depth 1 $LLVM_URL $LLVM_GIT_DIR || return
        else
            echo "-> git clone --depth 1 -b $LLVM_VERSION $LLVM_URL $LLVM_GIT_DIR"
            git clone --depth 1 -b $LLVM_VERSION $LLVM_URL $LLVM_GIT_DIR || return
        fi

        echo " ------  LLVM Cloned  ------ "

    fi

}
