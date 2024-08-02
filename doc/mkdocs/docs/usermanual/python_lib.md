# Using Shamrock as a python library

Aside from using Shamrock as a python interpreter for runscripts, it is also possible to use it as a python library and install it to the python distribution using pip.

This can be done in 3 steps:

- Activate a python virtual environment
- Create a Shamrock environment with the flag `--pylib`
- In the environment build directory run `` 

For exemple the following should work:
```bash
python -m venv .shamrock-venv
source .shamrock-venv/bin/activate
./env/new-env --machine debian-generic.acpp --builddir build_pylib --pylib -- --backend omp
cd build_pylib
pip install --verbose -e .
```

This create an editable library meaning that you change a source file and just rerun `pip install --verbose -e .` in the build directory to update the installation.