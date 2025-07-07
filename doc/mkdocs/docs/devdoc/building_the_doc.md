# Building the documentation

For the python doc, first have an environment active (`source ./activate`) where shamrock is compiled, then run:
```bash
python3 -m venv .pyvenv
source .pyvenv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
bash $SHAMROCK_DIR/doc/sphinx/gen_sphinx_doc.sh
```
