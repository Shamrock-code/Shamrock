#!/bin/bash

if ! (which python &> /dev/null || which python3 &> /dev/null); then
    echo "You need to have the python command available to generate the sphinx doc"
    exit 1
fi

if ! python3 -c "import shamrock" &> /dev/null; then
    echo "You need to have shamrock installed in your python path to generate the sphinx doc"
    exit 1
fi

pip_list=(
    "sphinx"
    "pydata-sphinx-theme"
    "sphinx-gallery"
    "memory-profiler"
    "sphinx-copybutton"
    "sphinx_design"
    )

for package in "${pip_list[@]}"; do
    if [ -z "$(pip list | grep $package)" ]; then
        echo "You need to have $package installed to generate the sphinx doc"
        echo "Running : pip install $package"
        pip install $package
    else
        echo "$package is installed."
    fi
done

set -e

cd "$(dirname "$0")"

make html

set +e

rm -rf examples/_to_trash
