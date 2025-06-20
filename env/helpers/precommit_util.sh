#!/bin/bash

export PRECOMMITVENV=$BUILD_DIR/.pyvenv_precommit

if [ ! -f "$PRECOMMITVENV/bin/activate" ]; then
    echo "No python venv were configured for the pre-commit"

    python3 -m venv $PRECOMMITVENV
    (source $PRECOMMITVENV/bin/activate && pip install pre-commit)
fi

function run_precommit {
    (source $PRECOMMITVENV/bin/activate && pre-commit)
}
