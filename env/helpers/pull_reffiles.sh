#!/bin/bash

export REF_FILES_PATH=$BUILD_DIR/reference-files

function pull_reffiles {
    git clone https://github.com/Shamrock-code/reference-files.git $REF_FILES_PATH
}
