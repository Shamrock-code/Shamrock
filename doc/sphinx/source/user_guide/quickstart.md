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

In this guide as you may have noticed above we are compiling using AdaptiveCpp with OpenMP backend so we can only run on CPU (device `2` in my case).

Now let's see if Shamrock can start on the selected device. To do so run

```bash
./shamrock --smi --sycl-cfg 2:2
```

It should add something like this

```
Selected devices : (totals can be wrong if using multiple ranks per device)
  - 1 x AdaptiveCpp OpenMP host device (id=2)
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
./shamrock --smi --sycl-cfg 2:2 --benchmark-mpi
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
./shamrock --smi --sycl-cfg 2:2 --benchmark-mpi --rscript ../examples/benchmarks/run_exclusive_scan_in_place.py
```

You should see a figure like:
![exscan perf figure](../../_images/sphx_glr_run_exclusive_scan_in_place_001.png)

:::{note}
Here the `--rscript` flag here means run-scripts. In Shamrock since everything goes through python your run will be a python script, hence the name "run script".
:::

### Python interpreter + Ipython

### Python package

### Jupyter notebook

### Legacy content for this guide

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
