#!/bin/bash
if [ -z ${ACPP_GIT_DIR+x} ]; then echo "ACPP_GIT_DIR is unset"; return; fi

if [ ! -f "$ACPP_GIT_DIR/README.md" ]; then
    echo " ------ Clonning AdaptiveCpp ------ "
    git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git $ACPP_GIT_DIR
    echo " ------  AdaptiveCpp Cloned  ------ "

fi
