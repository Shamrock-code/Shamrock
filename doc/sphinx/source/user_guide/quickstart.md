# Getting Started

## Installation

When it come to using Shamrock several options are available.

[![Packaging status](https://repology.org/badge/vertical-allrepos/shamrock.svg)](https://repology.org/project/shamrock/versions)

I will assume that you want to compile it from source (as most probably do anyway) in the following and precise what changes if you are installing it directly through other means. See the list for the alternatives:

- [Spack package](./quickstart/install_spack.md) (Easy but long compile time)
- [Homebrew package](./quickstart/install_brew.md) (Homebrew package, precompiled as well)
- [Docker container](./quickstart/install_docker.md) (Fastest but not the most convenient)

## Prerequisite

::::{tab-set}
:::{tab-item} Linux (Debian & Ubuntu)

If you don't have already have an llvm (...) :

```bash
wget --progress=bar:force https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20
sudo apt install -y libclang-20-dev clang-tools-20 libomp-20-dev
```

For the other requirements do :

```bash
sudo apt install cmake libboost-all-dev python3-ipython ninja-build
```

Just to ensure that you have the correct stuff do:

```bash
clang-20 --version
```

If you get an error that's weird and it is probably simpler to drop a message on [Discord](https://discord.gg/Q69s5buyr5) for help.

:::
:::{tab-item} MacOS

With Homebrew:

```bash
brew install cmake libomp boost open-mpi adaptivecpp python ninja
```

:::
:::{tab-item} ArchLinux

```bash
sudo pacman -Syu base-devel git python python-pip cmake boost ninja openmp openmpi doxygen llvm20 clang20 lld
```

:::
:::{tab-item} Conda

Nothing to do at this stage

:::
::::

## Doing the setup

### Cloning the repo

Now there before cloning the source code there are two options:

- Do you want to contribute stuff to Shamrock (e.g. modify it and propose the changes)
- Do you want to just use the standard version

::::{tab-set}
:::{tab-item} Use only

Go in the folder where you want to work and do:

```bash
git clone --recurse-submodules https://github.com/Shamrock-code/Shamrock.git
```

:::
:::{tab-item} Use and modify

This can be a bit more involved if you are not used to Github, but this is how to get work done there:

If you already have registered your SSH key on Github you don't need to touch it, otherwise:

- First go to [Github.com](https://github.com) and ensure that you are logged in.
- In a terminal on your laptop/desktop do `ssh-keygen -t rsa -b 4096` (I recommend rsa4096 since some supercomputer require it). You can leave the password empty if you want to avoid the need to type it. And you can also accept the default name of the key `id_rsa`.
- Now recover your public key `cat ~/.ssh/id_rsa.pub` (You may have to change the filename, this one is the default)
- Go to [Github SSH user key](https://github.com/settings/keys) and click on `New SSH Key`, chose a name and paste the key obtained by `cat ~/.ssh/id_rsa.pub` in the text box named `Key`.

Alright now that the SSH key is good:

- First go to [Github.com](https://github.com) and ensure that you are logged in.
- Go to the [Shamrock repo](https://github.com/Shamrock-code/Shamrock) and at the top right of the screen you should see a button called "Fork". Alternatively you can just go to that [URL](https://github.com/Shamrock-code/Shamrock/fork).
- And click on Create fork
- You should land on a page whose url is `https://github.com/<your github username>/Shamrock`

Now assuming you have registered your SSH key do:

```bash
git clone --recurse-submodules git@github.com:<your github username>/Shamrock.git
```

:::
::::

And go to the new folder

```bash
cd Shamrock
```

### Creating the environment

Shamrock provides its own utilities with pre-made configurations for various machines. Here i give recommendations adapted a quickstart guide. If you want more details about the environment setup see [This page](../user_guide/envs.md).

::::{tab-set}
:::{tab-item} Linux (Debian & Ubuntu)

```bash
./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp
```

:::
:::{tab-item} MacOS

```bash
./env/new-env --builddir build --machine macos-generic.acpp -- --backend omp
```

:::
:::{tab-item} ArchLinux

```bash
./env/new-env --machine archlinux.acpp --builddir build -- --backend omp
```

:::
:::{tab-item} Conda

```bash
./env/new-env --machine conda.acpp --builddir build -- --backend omp
```

:::
::::

### Compiling

```
# Now move in the build directory
cd build
# Activate the workspace, which will define some utility functions
source ./activate # load the correct modules and ENV vars
shamconfigure     # alias to the correct cmake command
shammake          # alias to ninja build (or make if ninja is not avail)
```

If you see any errors at this point it can be hard to list all cases so again drop a message on [Discord](https://discord.gg/Q69s5buyr5) for help.

## Starting Shamrock

:::{warning}
This guide assume that you have Shamrock compiled from source. I will mention what has to be changed if it is installed globally in notes in the following sections.
:::

You have 4 main ways of using Shamrock:

- As a Python interpreter
- As a Python interpreter + Ipython
- As a Python package
- In a Jupyter notebook

For all of them except the Python interpreter mode you should run the following beforehand:

```bash
# Configure install paths in the local build folder
cmake . -DCMAKE_INSTALL_PYTHONDIR=$(pwd)/pysham -DCMAKE_INSTALL_PREFIX=$(pwd)/shaminstall
# Install it !
shammake install
```

### Python interpreter mode

So assuming that you have Shamrock compiled and all we can start to do stuff with it. Let start simple just to check that the executable load correct:

```bash
./shamrock
```

:::{note}
If you have installed shamrock globally using any of the alternatives, just replace `./shamrock` by `shamrock` in this section
:::

You should see something like (even if the git infos & time might differ):

```
Warning: No kernel can be run without a sycl configuration (--sycl-cfg x:x)                                   [Init][rank=0]

  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░

Copyright (c) 2021-2026 Timothée David--Cléris (tim.shamrock@proton.me)
SPDX-License-Identifier : CeCILL Free Software License Agreement v2.1
Start time : 2026-04-16 22:27:18
-----------------------------------------------------

Shamrock version : 2025.10.0+git.8a0c0b85e.patch-2026-04-15-15-33.dirty

Git infos :
     commit : 8a0c0b85e3598ba83d124ffe8277bddf9ddc9430
     HEAD   : refs/heads/patch-2026-04-15-15-33, refs/remotes/origin/patch-2026-04-15-15-33
     modified files (since last commit):
        doc/sphinx/source/user_guide/quickstart.md

-----------------------------------------------------
-----------------------------------------------------
log status :
 - Loglevel: 0, enabled log types :
xxx: xxx ( logger::normal )
Warning: xxx ( logger::warn )                                                                                  [xxx][rank=0]
Error: xxx ( logger::err )                                                                                     [xxx][rank=0]
Warning: No sycl configuration (--sycl-cfg x:x) has been set, early exit                                      [Init][rank=0]
-----------------------------------------------------
 - MPI finalize
Exiting ...

 Hopefully it was quick :')
```

If it works great ! Next let's see what devices are available to perform the compute:

```bash
./shamrock --smi
```

You should see the same thing as before but with something like this added to it:

```
 ----- Shamrock SMI -----

Available devices :

1 x Shamrock process: ---------------------------------------------------------------------
| id |      Device name          |      Platform name     |  Type  |    Memsize   | units |
-------------------------------------------------------------------------------------------
|  0 |   NVIDIA GeForce RTX 3070 |      CUDA (platform 0) |    GPU |      7.63 GB |    46 |
|  1 |         Intel(R) Graphics |    OpenCL (platform 0) |    GPU |     58.00 GB |    64 |
|  2 |  AdaptiveCpp OpenMP h ... |    OpenMP (platform 0) |    CPU |     62.17 GB |    24 |
-------------------------------------------------------------------------------------------
```

If you are familiar with GPU you should recognize the similarity with commands such as `nvidia-smi`, `rocm-smi` or `xpu-smi`.

In this guide as you may have noticed above we are compiling using AdaptiveCpp with OpenMP backend so we can only run on CPU (device `2` in my case), for most peoples OpenMP tends to be device 0 so, that what i will use in the rest of the guide, do not forget to change it if that is not the case on your system like mine.

Now let's see if Shamrock can start on the selected device. To do so run

```bash
./shamrock --smi --sycl-cfg 0:0
```

It should add something like this

```
Selected devices : (totals can be wrong if using multiple ranks per device)
  - 1 x AdaptiveCpp OpenMP host device (id=0)
          - default_work_group_size = 1
          - global_mem_size = 62.17 GB
          - local_mem_size = No limit !
          - mem_base_addr_align = 8,
          - max_mem_alloc_size_dev = 16.00 EB,
          - max_mem_alloc_size_host = 62.17 GB,
          - pci_address = Unknown
      - Warnings:
          - mem_base_addr_align for is 8 bits. I will assume that this is an issue and default to 64 bits (8 bytes) instead.
  Total memory : 62.17 GB
  Total compute units : 24

-----------------------------------------------------
 - Code init: DONE.
```

Especially if you see `- Code init: DONE.` it means that Shamrock can execute compute kernels (which is arguably a pretty usefull thing XD). For the uninitiated a compute kernel is essentially a parallel section like a parallel for loop that was scheduled onto the device to perform computations. And in the GPU or modern CPU world a HPC code is a roughly just a large succesion of compute kernels.

Also if you want to see the performance of the hardware you are running add the `--benchmark-mpi` flag.

```bash
./shamrock --smi --sycl-cfg 0:0 --benchmark-mpi
```

For example on my desktop (at home very late at night when I'm writing this 😅) i get:

```
-----------------------------------------------------
Running micro benchmarks:
 - p2p bandwidth    : 15.2 GB.s^-1 (ranks : 0 -> 0) (loops : 1825)
 - saxpy (f32)   : 92 GB.s^-1 (min = 9.2e+10, max = 9.2e+10, avg = 9.2e+10) (2.2e+00 ms, 128.00 MB)
 - saxpy (f64)   : 83.8 GB.s^-1 (min = 8.4e+10, max = 8.4e+10, avg = 8.4e+10) (2.4e+00 ms, 128.00 MB)
 - saxpy (f32_2)   : 89.3 GB.s^-1 (min = 8.9e+10, max = 8.9e+10, avg = 8.9e+10) (2.3e+00 ms, 128.00 MB)
 - saxpy (f64_2)   : 75.2 GB.s^-1 (min = 7.5e+10, max = 7.5e+10, avg = 7.5e+10) (2.7e+00 ms, 128.00 MB)
 - saxpy (f32_3)   : 98.1 GB.s^-1 (min = 9.8e+10, max = 9.8e+10, avg = 9.8e+10) (2.1e+00 ms, 128.00 MB)
 - saxpy (f64_3)   : 95.7 GB.s^-1 (min = 9.6e+10, max = 9.6e+10, avg = 9.6e+10) (2.1e+00 ms, 128.00 MB)
 - saxpy (f32_4)   : 85.9 GB.s^-1 (min = 8.6e+10, max = 8.6e+10, avg = 8.6e+10) (1.2e+00 ms, 64.00 MB)
 - saxpy (f64_4)   : 74.4 GB.s^-1 (min = 7.4e+10, max = 7.4e+10, avg = 7.4e+10) (2.7e+00 ms, 128.00 MB)
 - fma_chains (f32) : 40.6 Gflops (min = 4.1e+10, max = 4.1e+10, avg = 4.1e+10) (2.1e+02 ms, rotations = 64)
 - fma_chains (f64) : 38.1 Gflops (min = 3.8e+10, max = 3.8e+10, avg = 3.8e+10) (4.5e+02 ms, rotations = 128)
 - fma_chains (f32_2) : 81.8 Gflops (min = 8.2e+10, max = 8.2e+10, avg = 8.2e+10) (2.1e+02 ms, rotations = 64)
 - fma_chains (f64_2) : 65 Gflops (min = 6.5e+10, max = 6.5e+10, avg = 6.5e+10) (2.6e+02 ms, rotations = 64)
 - fma_chains (f32_3) : 108 Gflops (min = 1.1e+11, max = 1.1e+11, avg = 1.1e+11) (2.4e+02 ms, rotations = 64)
 - fma_chains (f64_3) : 100 Gflops (min = 1.0e+11, max = 1.0e+11, avg = 1.0e+11) (5.1e+02 ms, rotations = 128)
 - fma_chains (f32_4) : 160 Gflops (min = 1.6e+11, max = 1.6e+11, avg = 1.6e+11) (2.2e+02 ms, rotations = 64)
 - fma_chains (f64_4) : 119 Gflops (min = 1.2e+11, max = 1.2e+11, avg = 1.2e+11) (2.9e+02 ms, rotations = 64)
 - vector_allgather (u64, n=   1) : 3.467e-06 s (min = 3.47e-06, max = 3.47e-06, loops = 28841)
 - vector_allgather (u64, n=   8) : 3.431e-06 s (min = 3.43e-06, max = 3.43e-06, loops = 29146)
 - vector_allgather (u64, n=  64) : 3.451e-06 s (min = 3.45e-06, max = 3.45e-06, loops = 28978)
 - vector_allgather (u64, n= 128) : 3.384e-06 s (min = 3.38e-06, max = 3.38e-06, loops = 29554)
 - vector_allgather (u64, n= 150) : 3.363e-06 s (min = 3.36e-06, max = 3.36e-06, loops = 29733)
 - vector_allgather (u64, n=1024) : 3.392e-06 s (min = 3.39e-06, max = 3.39e-06, loops = 29484)
-----------------------------------------------------
```

Alright ! Let's run something "usefull" (we will run actual simulation in the next part of the tutorial), it will be one of the algorithm benchmarks.

```bash
./shamrock --smi --sycl-cfg 0:0 --rscript ../examples/benchmarks/run_exclusive_scan_in_place.py
```

You should see a figure like:
![exscan perf figure](../../_images/sphx_glr_run_exclusive_scan_in_place_001.png)

:::{note}
Here the `--rscript` flag here means run-scripts. In Shamrock since everything goes through python your run will be a python script, hence the name "run script". Here it is a benchmark of one of Shamrock's algorithms.
:::

### Python interpreter + Ipython

:::::{warning}
IPython does not come by default with python. Ensure that it is installed on you system, or you can install it in a python venv but then things are a bit different, see [Using Shamrock with Python venv](./python_venv.md).

::::{tab-set}
:::{tab-item} Linux (Debian & Ubuntu)

```bash
sudo apt install python3-ipython
```

:::
:::{tab-item} MacOS

```bash
brew install ipython
```

:::
:::{tab-item} ArchLinux

```bash
sudo pacman -Syu ipython
```

:::
:::{tab-item} Conda

TODO: we should add it to the conda env probably by default to avoid the need for the user to tweak it.

:::
::::

:::::

Alright! We can do interactive stuff now by using the `--ipython` flag in-place of the `--rscript` flag. Just know that this is limited to use with a single process. If you want to use Shamrock with MPI to distribute across multiple machines forget about interactive modes 😅.

```bash
./shamrock --smi --sycl-cfg 0:0 --ipython
```

Which should display something like:

```ipython
--------------------------------------------
-------------- ipython ---------------------
--------------------------------------------
SHAMROCK Ipython terminal
Python 3.14.3 (main, Feb 13 2026, 15:31:44) [GCC 15.2.1 20260209]

###
import shamrock
###

In [1]:
```

Now you can use the same python as one would in runscripts. A classic one to run there is the following (which is what i do in the basic CI test btw):

```{code-block} python
---
lineno-start: 1
---
import shamrock
print("Shamrock version:", shamrock.version_string())
print("Git info:", shamrock.get_git_info())
shamrock.change_loglevel(1)
if not shamrock.sys.is_initialized():
    shamrock.sys.init('0:0')

# To test that importing nested modules works
from shamrock.math import *
```

You should get something like (do not forget to change the device selection in `shamrock.sys.init('0:0')` if you are not using device 0):

```python
--------------------------------------------
-------------- ipython ---------------------
--------------------------------------------
SHAMROCK Ipython terminal
Python 3.14.3 (main, Feb 13 2026, 15:31:44) [GCC 15.2.1 20260209]

###
import shamrock
###

In [1]: import shamrock
   ...: print("Shamrock version:", shamrock.version_string())
   ...: print("Git info:", shamrock.get_git_info())
   ...: shamrock.change_loglevel(1)
   ...: if not shamrock.sys.is_initialized():
   ...:     shamrock.sys.init('0:0')
   ...:
   ...: # To test that importing nested modules works
   ...: from shamrock.math import *
Shamrock version: 2025.10.0+git.bb2b3814a.patch-2026-04-15-15-33.dirty
Git info:      commit : bb2b3814af0ec0c2f16f46017185dbdba36d21fa
     HEAD   : refs/heads/patch-2026-04-15-15-33, refs/remotes/origin/patch-2026-04-15-15-33
     modified files (since last commit):
        doc/sphinx/source/user_guide/quickstart.md

-> modified loglevel to 0 enabled log types :
log status :
 - Loglevel: 1, enabled log types :
Info: xxx ( logger::info )                                                                [xxx][rank=0]
xxx: xxx ( logger::normal )
Warning: xxx ( logger::warn )                                                             [xxx][rank=0]
Error: xxx ( logger::err )                                                                [xxx][rank=0]

In [2]:
```

If you want to run the same script as above just do (in the IPython prompt):

```ipython
run ../examples/benchmarks/run_exclusive_scan_in_place.py
```

### Python package

This one is simple. First ensure that you have done:

```bash
# Configure install paths in the local build folder
cmake . -DCMAKE_INSTALL_PYTHONDIR=$(pwd)/pysham -DCMAKE_INSTALL_PREFIX=$(pwd)/shaminstall
# Install it !
shammake install
```

Now you just need to help python locate Shamrock python bindings which are in `./pysham`. To do that just prepend the python call by extending the python path like so

```bash
PYTHONPATH=./pysham:$PYTHONPATH python3 # can be just python on some system
```

Then use the same thing as above to test it:

```py
❯ PYTHONPATH=./pysham:$PYTHONPATH python3
Python 3.14.3 (main, Feb 13 2026, 15:31:44) [GCC 15.2.1 20260209] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import shamrock
... print("Shamrock version:", shamrock.version_string())
... print("Git info:", shamrock.get_git_info())
... shamrock.change_loglevel(1)
... if not shamrock.sys.is_initialized():
...     shamrock.sys.init('0:0')
...
... # To test that importing nested modules works
... from shamrock.math import *
...
Shamrock version: 2025.10.0+git.bb2b3814a.patch-2026-04-15-15-33.dirty
Git info:      commit : bb2b3814af0ec0c2f16f46017185dbdba36d21fa
     HEAD   : refs/heads/patch-2026-04-15-15-33, refs/remotes/origin/patch-2026-04-15-15-33
     modified files (since last commit):
        doc/sphinx/source/user_guide/quickstart.md

-> modified loglevel to 0 enabled log types :
log status :
 - Loglevel: 1, enabled log types :
[xxx] Info: xxx ( logger::info )
[xxx] : xxx ( logger::normal )
[xxx] Warning: xxx ( logger::warn )
[xxx] Error: xxx ( logger::err )
>>>
```

As you can we have just imported and initialize Shamrock as a python package. Again to run the same script (for the third time already) you can do:

```bash
PYTHONPATH=./pysham:$PYTHONPATH python3 ../examples/benchmarks/run_exclusive_scan_in_place.py
```

### Jupyter notebook

Just for your pleasure and enjoyment through this flight... tutorial behold the magic command:

```bash
(
    python3 -m venv .pyvenv
    source .pyvenv/bin/activate
    pip install -U notebook
    export PYTHONPATH=./pysham:$PYTHONPATH
    jupyter notebook
)
```

It leverages the `export PYTHONPATH` to register the shamrock modules in the python venv even though Shamrock was never installed in it. Anyway thanks to that the ipython kernel launched inside jupyter should find Shamrock gracefully. Also note the parenthesis around it. This ensure that the modifications made to the environment variables of your current terminal are contained only to the scope of this sequence of command and everything goes back to normal after.

:::{note}
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
- `--no-browser` Do not open the browser there are none in the container obviously.
- `--ip=0.0.0.0` Otherwise the port is not fowarded correctly out of the container.
- `--NotebookApp.token=''` Do not use a token to log.
:::

## Next steps

Alright now that you know how to compile & start Shamrock we can go many paths:

- [Use Shamrock with python venv](./python_venv.md)
- [Use Shamrock on GPU](./use_on_gpu.md)

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart/install_spack.md
quickstart/install_brew.md
quickstart/install_docker.md
quickstart/install_from_source.md
quickstart/recommended_config/linux_debian.md
quickstart/recommended_config/macos.md
quickstart/recommended_config/conda.md
```
