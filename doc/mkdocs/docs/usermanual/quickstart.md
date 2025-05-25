# Getting Started

## Installation

There are multiple ways to install Shamrock:

 - [From source](./quickstart/install_from_source.md)
 - [Spack package](./quickstart/install_spack.md) (Easy but long compile time)
 - [Homebrew package](./quickstart/install_brew.md) (Homebrew package, precompiled as well)
 - [Docker container](./quickstart/install_docker.md) (Fastest but not the most convenient)


## Starting Shamrock


!!! warning

    This guide assume that shamrock is available in your path (aka that the `shamrock` commands and that `python3 -c "import shamrock"` work). This might not be the case if you have installed Shamrock [from source](./quickstart/install_from_source.md).

    In that case you have two options:

    - export the paths (assuming you are in the build directory) like so :
    ```bash
    export PATH=$(pwd):$PATH
    export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

    - Or precede shamrock start with the right env variable (assuming you are in the build directory) :
    ```bash
    ./shamrock <...>
    PYTHONPATH=$(pwd):$PYTHONPATH python3 <...>
    ```

You have 4 main ways of using Shamrock:

 - As Ipython mode
 - As a python interpreter
 - As a Python package
 - In a jupyter notebook

## Selecting the device to run on

To select the device that you want to run on run the following command:

```bash
shamrock --smi
```

You should see something like:
```
 ----- Shamrock SMI -----

Available devices :

1 x Nodes: --------------------------------------------------------------------------------
| id |      Device name          |      Platform name     |  Type  |    Memsize   | units |
-------------------------------------------------------------------------------------------
|  0 |   NVIDIA GeForce RTX 3070 |      CUDA (platform 0) |    GPU |      7.63 GB |    46 |
|  1 |  AdaptiveCpp OpenMP h ... |    OpenMP (platform 0) |    CPU |     62.18 GB |    24 |
-------------------------------------------------------------------------------------------
```

Use the one you prefer by passing its id to the `--sycl-cfg x:x` flag like so

```bash
shamrock --smi --sycl-cfg 0:0
```

You should see :
```
 ----- Shamrock SMI -----

Available devices :

1 x Nodes: --------------------------------------------------------------------------------
| id |      Device name          |      Platform name     |  Type  |    Memsize   | units |
-------------------------------------------------------------------------------------------
|  0 |   NVIDIA GeForce RTX 3070 |      CUDA (platform 0) |    GPU |      7.63 GB |    46 |
|  1 |  AdaptiveCpp OpenMP h ... |    OpenMP (platform 0) |    CPU |     62.18 GB |    24 |
-------------------------------------------------------------------------------------------

Selected devices : (totals can be wrong if using multiple ranks per device)
  - 1 x NVIDIA GeForce RTX 3070 (id=0)
  Total memory : 7.63 GB
  Total compute units : 46
```

If you quickly want to test that everything works do:

```bash
shamrock --smi --sycl-cfg 0:0 --benchmark-mpi
```

You should get:
```
 ----- Shamrock SMI -----

Available devices :

1 x Nodes: --------------------------------------------------------------------------------
| id |      Device name          |      Platform name     |  Type  |    Memsize   | units |
-------------------------------------------------------------------------------------------
|  0 |   NVIDIA GeForce RTX 3070 |      CUDA (platform 0) |    GPU |      7.63 GB |    46 |
|  1 |  AdaptiveCpp OpenMP h ... |    OpenMP (platform 0) |    CPU |     62.18 GB |    24 |
-------------------------------------------------------------------------------------------

Selected devices : (totals can be wrong if using multiple ranks per device)
  - 1 x NVIDIA GeForce RTX 3070 (id=0)
  Total memory : 7.63 GB
  Total compute units : 46

-----------------------------------------------------
Running micro benchmarks:
 - p2p bandwidth    : 2.4662e+10 B.s^-1 (ranks : 0 -> 0) (loops : 2969)
 - saxpy (f32_4)   : 4.005e+11 B.s^-1 (min = 4.0e+11, max = 4.0e+11, avg = 4.0e+11) (2.0e+00 ms)
 - add_mul (f32_4) : 1.340e+13 flops (min = 1.3e+13, max = 1.3e+13, avg = 1.3e+13) (1.9e+01 ms)
 - add_mul (f64_4) : 2.265e+11 flops (min = 2.3e+11, max = 2.3e+11, avg = 2.3e+11) (1.1e+03 ms)
```

Here you can check that the peak flop & bandwidth match somewhat to the spec of your device.

!!! warning
    It is normal if the flops are off by about a factor two, the add_mul benchmark only targets
    `add` and `mul` instructions which do not stress the full floating point units.

## Using the Ipython mode

Before using the Ipython mode check that you have Ipython installed otherwise you will get an error.

To use the Ipython mode do:
```bash
shamrock --sycl-cfg 0:0 --ipython
```

At the end of the ouput you should be prompted with a Ipython terminal:
```py
--------------------------------------------
-------------- ipython ---------------------
--------------------------------------------
SHAMROCK Ipython terminal
Python 3.12.9 (main, Feb  4 2025, 14:38:38) [GCC 14.2.1 20241116]

###
import shamrock
###

In [1]: import shamrock
```

After this you can use the shamrock python package like you would normally ([:octicons-arrow-right-24: Python frontend documentation](../../sphinx/index.html)).


## Using the Python interpreter mode

You can also use Shamrock to run python scripts. For example let's say that we have the following python file:
```py linenums="1" title="test.py"
import shamrock

# If you are using the shamrock executable the init is handled before starting python.
# Hence this will be skipped, if you are using the python package this will take care of the init.
if not shamrock.sys.is_initialized():
    shamrock.sys.init("0:0")

shamrock.change_loglevel(1) # change loglevel to level 1 (info)
print(shamrock.get_git_info())
```

You can use Shamrock to execute it using the `--rscript` flag (rscript stands for runscript).
```
shamrock --sycl-cfg 0:0 --rscript test.py
```

You will get something like:
```
-----------------------------------
running pyscript : test.py
-----------------------------------
-> modified loglevel to 1, enabled log types :
Info: xxx ( logger::info )                                       [xxx][rank=0]
xxx: xxx ( logger::normal )
Warning: xxx ( logger::warn )                                    [xxx][rank=0]
Error: xxx ( logger::err )                                       [xxx][rank=0]

     commit : 966110450587dd1f0ed53438e181835d39004650
     HEAD   : refs/heads/update-readme
     modified files (since last commit):
        README.md
        doc/mkdocs/docs/usermanual/quickstart.md
        doc/mkdocs/mkdocs.yml
        exemples/godunov_sod.py
        src/main.cpp
        src/shamsys/include/shamsys/NodeInstance.hpp
        src/shamsys/src/NodeInstance.cpp

-----------------------------------
pyscript end
-----------------------------------
```

## Using Shamrock as a python package

```py linenums="1" title="test.py"
import shamrock

# If you are using the shamrock executable the init is handled before starting python.
# Hence this will be skipped, if you are using the python package this will take care of the init.
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

shamrock.sys.close()
```

Then you can simply run it:
```bash
python3 test.py
```

You should get the following :
```
-> modified loglevel to 0 enabled log types :
[xxx] Info: xxx ( logger::info )
[xxx] : xxx ( logger::normal )
[xxx] Warning: xxx ( logger::warn )
[xxx] Error: xxx ( logger::err )

-----------------------------------------------------
 - MPI finalize
Exiting ...

 Hopefully it was quick :')

```

## Why do we have to distinguish the init between the python package and executable ?

Initializing MPI and GPU software stacks can be complex and inconsistent across platforms (yeah I know it sucks ...).
It may happen that some shared libraries are loaded by Python, potentially disrupting the MPI initialization.
For instance, importing a package in Python that utilizes CUDA may load the CUDA stubs before Shamrock initializes,
which could mess up MPI direct GPU communication.

To prevent such edge cases, the executable mode initializes all required components before starting Python,
ensuring the proper initialization of the libraries.

## Jupyter notebook mode

I will assume here that you have jupyter installed in the same python distribution
(If not try to use you system package manager to install it, or in last resort using pip).

You can then run `jupyter notebook` which will start it.
Make then sure that you select the python kernel that match the python distribution where Shamrock is installed.

If you want to check that try this command:
```
echo "import sys;print(sys.executable)" > test.py && shamrock --sycl-cfg 0:0 --rscript test.py
```
It will print the corresponding python executable :
```
-----------------------------------
running pyscript : test.py
-----------------------------------
/usr/bin/python3
-----------------------------------
pyscript end
-----------------------------------
```

!!! note
    If you are using the Shamrock Docker container there is a special case here.
    Instead to run jupyter, do the following

    ```bash
    docker run -i -t -v $(pwd):/work -p 8888:8888 --platform=linux/amd64 ghcr.io/shamrock-code/shamrock:latest-oneapi jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --NotebookApp.token=''
    ```

    This will start jupyter in the current folder, you can then go to <http://127.0.0.1:8888/> to use it.

    Explanation of the flags:

     - `-i` start the docker container in interactive mode.
     - `-t` start a terminal.
     - `-v $(pwd):/work` mount the current working directory to `/work` in the docker container.
     - `-p 8888:8888` forward the 8888 port from inside the container.
     - `--platform=linux/amd64` If you are on macos this will start a virtual machine.
     - `ghcr.io/shamrock-code/shamrock:latest-oneapi` the docker container.
     - `jupyter notebook` Come on you know what this does, do you ???
     - `--allow-root` Inside the docker container you are root so you should bypass this check.
     - `--no-browser` Do not open the browser there are not in the container obviously.
     - `--ip=0.0.0.0` Otherwise the port is not fowarded correctly out of the container.
     - `--NotebookApp.token=''` Do not use a token to log.
