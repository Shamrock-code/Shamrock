# Using Shamrock as a python library

Aside from using Shamrock as a python interpreter for runscripts, it is also possible to use it as a python library and install it to the python distribution using pip.

This can be done in 3 steps:

- Activate a python virtual environment
- Create a Shamrock environment
- In the environment build directory run ``

For exemple the following should work:
```bash
python -m venv .shamrock-venv
source .shamrock-venv/bin/activate
./env/new-env --machine debian-generic.acpp --builddir build_pylib -- --backend omp
cd build_pylib
pip install --verbose -e .
```

This create an editable library meaning that you change a source file and just rerun `pip install --verbose -e .` in the build directory to update the installation.

## Jupyter notebook

Jupyter notebooks are great ... However, for odd reasons starting a jupyter notebook in a python env
does not start the python kernel of the python venv, so there are additional steps ....

So you should run this in your venv:
```bash
pip install ipython matplotlib numpy jupyter ipykernel
python -m ipykernel install --user --name=shamrock_venv
```

Then start a jupyter notebook and select `shamrock_venv` as python kernel.

Another alternative is to tun this command instead, but it seems to install the kernel globally, so I do not really recommend it.
```bash
ipython kernel install --user --name=venv_shamrock
```
