#!/usr/bin/env bash

cd "$(dirname "$0")"

EXAMPLE_FILE="$1"

echo "Moving the example file to .tmp :"
mv ${EXAMPLE_FILE} .tmp

bash gen_sphinx_doc_single_example.sh do_not_run_annything_dammit

snapshot() {
  find . -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
}

echo "Snapshotting the current directory :"
snapshot > /tmp/before.sha


echo "Moving the example file back to the original location :"
mv .tmp ${EXAMPLE_FILE}

echo "Generating the sphinx doc for the example :"
bash gen_sphinx_doc_single_example.sh "${EXAMPLE_FILE}"

echo "Snapshotting the current directory :"
snapshot > /tmp/after.sha

echo "Diffing the snapshots :"

# The only think AI is good for is to generate regex BS
awk 'NR==FNR {a[$2]=$1; next}
     !($2 in a) || a[$2] != $1 {print $2}' \
     /tmp/before.sha /tmp/after.sha > /tmp/diff

# print the list of files that changed
echo "Files that changed :"
cat /tmp/diff

# Keep only files listed in /tmp/diff inside source/ and examples/
for dir in source examples; do
  find "$dir" -type f | while read -r file; do
    # Add ./ prefix to match the format in /tmp/diff
    if ! grep -Fxq "./$file" /tmp/diff; then
      echo "Removing $file"
      rm -f "$file"
    else
      echo "Keeping $file"
    fi
  done
done
echo "Removing build directory"
rm -rf build

echo "Tree of the current directory :"
tree .

set +e

rm -rf examples/_to_trash
