#!/usr/bin/env bash

PYMKDOCS="$1"
echo "Using Python executable: ${PYMKDOCS}"

export VENV_DIR=.pyenv

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No python venv were configured for the doc"
    $PYMKDOCS -m venv $VENV_DIR
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
    source $VENV_DIR/bin/activate && mkdocs build
)
