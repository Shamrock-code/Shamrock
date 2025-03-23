#!/bin/bash

function clone_acpp {

    ACPP_URL="https://github.com/AdaptiveCpp/AdaptiveCpp.git"
    
    if [ -z ${ACPP_GIT_DIR+x} ]; then echo "ACPP_GIT_DIR is unset"; return 1; fi

    if [ ! -f "$ACPP_GIT_DIR/README.md" ]; then
        echo " ------ Clonning AdaptiveCpp ------ "

        if [ -z ${ACPP_VERSION+x} ]
        then
            echo "-> git clone $ACPP_URL $ACPP_GIT_DIR"
            git clone $ACPP_URL $ACPP_GIT_DIR  || return
        else
            echo "-> git clone -b $ACPP_VERSION $ACPP_URL $ACPP_GIT_DIR"
            git clone -b $ACPP_VERSION $ACPP_URL $ACPP_GIT_DIR || return
        fi

        echo " ------  AdaptiveCpp Cloned  ------ "

    fi
 
}
