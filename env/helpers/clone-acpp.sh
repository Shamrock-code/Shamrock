#!/bin/bash
if [ -z ${ACPP_GIT_DIR+x} ]; then echo "ACPP_GIT_DIR is unset"; return; fi

if [ ! -f "$ACPP_GIT_DIR/README.md" ]; then
    echo " ------ Clonning AdaptiveCpp ------ "

    if [ -z ${ACPP_VERSION+x} ]
    then
        echo "-> git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git $ACPP_GIT_DIR"
        git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git $ACPP_GIT_DIR
    else
        echo "-> git clone -b $ACPP_VERSION https://github.com/AdaptiveCpp/AdaptiveCpp.git $ACPP_GIT_DIR"
        git clone -b $ACPP_VERSION https://github.com/AdaptiveCpp/AdaptiveCpp.git $ACPP_GIT_DIR
    fi

    echo " ------  AdaptiveCpp Cloned  ------ "

fi
