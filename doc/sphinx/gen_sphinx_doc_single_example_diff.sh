#!/usr/bin/env bash

cd "$(dirname "$0")"

EXAMPLE_FILE="$1"

bash gen_sphinx_doc_single_example.sh do_not_run_annything_dammit

snapshot > /tmp/before.sha
bash gen_sphinx_doc_single_example.sh "${EXAMPLE_FILE}"
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
