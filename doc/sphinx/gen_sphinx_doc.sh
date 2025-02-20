#!/bin/bash

if ! (which python &> /dev/null || which python3 &> /dev/null); then
    echo "You need to have the python command available to generate the sphinx doc"
    exit 1
fi

if ! python3 -c "import shamrock" &> /dev/null; then
    echo "You need to have shamrock installed in your python path to generate the sphinx doc"
    exit 1
fi

if [ -z "$(pip list | grep Sphinx)" ]; then
    echo "You need to have sphinx installed to generate the sphinx doc"
    echo "Running : pip install sphinx"
    pip install sphinx
fi

if [ -z "$(pip list | grep pydata-sphinx-theme)" ]; then
    echo "You need to have pydata-sphinx-theme installed to generate the sphinx doc"
    echo "Running : pip install pydata-sphinx-theme"
    pip install pydata-sphinx-theme
fi

make html
