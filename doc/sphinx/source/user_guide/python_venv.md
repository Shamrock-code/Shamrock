# Using Shamrock with Python venv

Alright, so now that you (in theory) went through the quickstart, you may wonder how to use Python venv to use Python packages without being forced to install them globally on your system.

## Creating the venv and installing basic utilities

Ok, first off create the actual venv. It's personal taste, but I tend to like naming it `.pyvenv`; you are free to name it whatever you prefer. Anyway, just do:

```bash
python3 -m venv .pyvenv
```

It will create a `.pyvenv` folder, which is the environment. Now you can activate it (notice the similarity with Shamrock envs):

```bash
source .pyvenv/bin/activate
```

It depends a bit on the shell you are using, but there will likely be an indicator like `(.pyvenv)` or something like that that will appear in the shell console.

Now if you run `pip list` you will notice that it's very bare (same if you run `python` and import matplotlib, for example). Since you are in a venv, you can install anything without altering your system install. I recommend the following list (which is used in the docs).

```bash
pip install -U matplotlib numpy scipy sphinx sympy numba ipython
```

Now you can do whatever you want with it, it is configured.

## Use it with Shamrock

If you run the `shamrock` executable, you should set the flag `--pypath-from-bin python` to use the Python corresponding to the command `python`, which here is the venv (yeah that's a lot of `python` on a single line 😂).

::::{tab-set}
:::{tab-item} Interpreter mode

```bash
./shamrock --smi --sycl-cfg 0:0 --pypath-from-bin python --rscript <the path to your .py script>
```

:::
:::{tab-item} Ipython mode

```bash
./shamrock --smi --sycl-cfg 0:0 --pypath-from-bin python --ipython
```

:::
::::

:::{note}
I'm still not sure if I like the name of that flag `--pypath-from-bin`; it is still open to discussion.
:::

If you are using Shamrock as a Python package, it does not change (except that you can type `python` instead of `python3`):

```bash
PYTHONPATH=./pysham:$PYTHONPATH python <the path to your .py script>
```

For Jupyter you already have a venv so just do:

```bash
(
    pip install -U notebook
    export PYTHONPATH=./pysham:$PYTHONPATH
    jupyter notebook
)
```
