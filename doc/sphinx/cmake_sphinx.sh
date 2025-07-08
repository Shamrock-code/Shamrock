#!/bin/bash

PYSPHINX="$1"
echo "Using Python executable: ${PYSPHINX}"

export VENV_DIR=.sphinxvenv

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No python venv were configured for the doc"
    $PYSPHINX -m venv $VENV_DIR
fi

(
    source $VENV_DIR/bin/activate && \
    pip install -U \
        matplotlib \
        numpy \
        scipy \
        sphinx \
        pydata-sphinx-theme \
        sphinx-gallery \
        memory-profiler \
        sphinx-copybutton \
        sphinx_design
)

PYTHONPATH=$(pwd):$PYTHONPATH $SHAMROCK_DIR/doc/sphinx/gen_sphinx_doc.sh
