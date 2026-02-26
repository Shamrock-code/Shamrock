#!/usr/bin/env bash

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
    "sphinxcontrib-video"
    "sympy"
    "matplotlib"
    "numpy"
    "scipy"
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

# Autodetect number of CPU cores, fallback to 1 if nproc is not available
PARALLEL_JOBS=$(nproc 2>/dev/null || echo "1")
echo "Using parallel jobs: ${PARALLEL_JOBS}"

make html SPHINXOPTS="-j ${PARALLEL_JOBS}"

set +e

rm -rf examples/_to_trash
