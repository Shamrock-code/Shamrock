# Shamrock install (From source)

## Shamrock Download

You have two options: Using your fork of shamrock,Using the main repo

```bash
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git
```
If you want to use your fork clone it using (replace `<login>` by your github username):
```bash
git clone --recurse-submodules git@github.com:<login>/Shamrock.git
```

One of the easiest way to get started is to use the Shamrock environments which setup SYCL compiler and configure/compile shamrock for you.

## AdaptiveCpp setup

First start by checking that you have the right packages installed on your system

!!! info end "Installation instruction"
    To install recommanded package to compile AdaptiveCpp
    === "Linux (debian)"

        If you don't have llvm (...) :

        ```bash
        wget --progress=bar:force https://apt.llvm.org/llvm.sh
        chmod +x llvm.sh
        sudo ./llvm.sh 18
        sudo apt install -y libclang-18-dev clang-tools-18 libomp-18-dev
        ```

        for the other requirements :
        ```bash
        sudo apt install cmake libboost-all-dev python3-ipython
        ```

    === "MacOS"

        ```bash
        brew install cmake libomp boost open-mpi adaptivecpp
        ```

    === "Conda"

        Nothing to do at this stage

## First use of the environments

Now you can initialise a Shamrock environment:

=== "Linux (debian)"

    ```bash
    ./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp
    ```

=== "MacOS"

    ```bash
    ./env/new-env --builddir build --machine macos-generic.acpp -- --backend omp
    ```
=== "Conda"

    ```bash
    ./env/new-env --machine conda.acpp --builddir build -- --backend omp
    ```

And then to configure & compile Shamrock:
```bash
# Now move in the build directory
cd build
# Activate the workspace, which will define some utility functions
source ./activate
# Configure Shamrock
shamconfigure
# Build Shamrock
shammake
```

## Wait ??? What happened ?

The first step is the `new-env` script which given a build folder and a machine (describe a system to compile on) will setup the build directory with everything required to compile Shamrock. Typically the machine argument allows specializing the configuration to specific systems (eg. `debian-generic.acpp`, `debian-generic.intel-llvm`, `adastra-mi250x.intel-llvm`, ...). The end goal of the environment is to setup all the exports, or modules loads on supercomputers through a common script.

When running the first step it will print something close to :
```
> ./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp
loading : Debian generic AdaptiveCpp
------------------------------------------
Running env setup for : Debian generic AdaptiveCpp
------------------------------------------
-- setting acpp target to : omp
-- generator not specified, defaulting to : ninja
-- clonning https://github.com/AdaptiveCpp/AdaptiveCpp.git
```
And then it will go on configuring & compiling AdaptiveCpp and give you a handy little script ... `activate` available in the build folder.

Now move in the build directory :
```sh
cd build
```

Activate the workspace, which will define some utility bash functions:
```sh
source ./activate
```

For exemple you have access to:

 - `setupcompiler` : which setup the compiler
 - `updatecompiler` : which update the compiler then recompile it
 - `shamconfigure` : configure shamrock
 - `shammake` : build Shamrock

The cool thing is that the Cmake command to build Shamrock can get quite complex for some configuration, the use of environments especially the `shamconfigure` command completely hides it. When a new supercomputer appears we just have to write a environment script for it, and then its done.

## Check that everything is running fine

Does the executable start ?
```
./shamrock --help
```
or
```
./shamrock_test --help
```

Both commands should just print the help.

## Quick fixes
### Error while loading shared libraries
If you get something like :
```
> ./shamrock
./shamrock: error while loading shared libraries: libsycl.so.7: cannot open shared object file: No such file or directory
```

You have missing libraries in the path, you can check that this is the case using the following command :
```bash
ldd shamrock | grep "not found"
```

if any libraries are not in the path it will print something like:
```
> ldd shamrock | grep "not found"
    libsycl.so.7 => not found
    libsycl.so.7 => not found
    libsycl.so.7 => not found
    libsycl.so.7 => not found
    libsycl.so.7 => not found
```

just do `export LD_LIBRARY_PATH=<path to lib>:$LD_LIBRARY_PATH` with `<path to lib>`
replaced by the path to the missing libraries.

## Remarks

If you want to setup the code with GPUs or do any advanced configuration please heads to those guides : ...
