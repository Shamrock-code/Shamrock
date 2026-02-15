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

EXAMPLE_FILE="$1"
echo "Using example file: ${EXAMPLE_FILE}"

snapshot() {
  find . -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
}

make html SPHINXOPTS="-D sphinx_gallery_conf.filename_pattern=do_not_run_annything_dammit"

snapshot > /tmp/before.sha
make html SPHINXOPTS="-D sphinx_gallery_conf.filename_pattern=${EXAMPLE_FILE}"
snapshot > /tmp/after.sha

echo "Diffing the snapshots :"

# The only think AI is good for is to generate regex BS
awk 'NR==FNR {a[$2]=$1; next}
     !($2 in a) || a[$2] != $1 {print $2}' \
     /tmp/before.sha /tmp/after.sha > /tmp/diff

# print the list of files that changed
echo "Files that changed :"
cat /tmp/diff

# tar the changed files
echo "Tarring the changed files :"
tar -cvf /tmp/changed_files.tar -T /tmp/diff

set +e

rm -rf examples/_to_trash
