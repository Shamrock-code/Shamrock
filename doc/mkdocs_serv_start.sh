#!/bin/bash

#if running in bash
if [ -n "$BASH_VERSION" ]; then
    CURRENT_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
else
    echo $(dirname "$0")
    CURRENT_DIR=$(realpath $(dirname "$0"))
fi

echo "Documentation dir is : $CURRENT_DIR"

cd $CURRENT_DIR/mkdocs

export VENV_DIR=.pyenv

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No python venv were configured for the doc"
    python3 -m venv $VENV_DIR
fi

(
    source $VENV_DIR/bin/activate && \
    pip install -U \
        matplotlib \
        numpy \
        scipy \
        mkdocs-material \
        mkdocs-git-committers-plugin-2 \
        mkdocs-git-authors-plugin \
        mkdocs-git-revision-date-plugin mkdocs-git-revision-date-localized-plugin \
)

(
    source $VENV_DIR/bin/activate && mkdocs serve
)
