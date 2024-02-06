# Getting Started

## Download the Workspace

One of the easiest way to get started is to use the ShamrockWorkspace which setup SYCL compiler and configure shamrock for you.
To start clone the repo :
```bash
git clone https://github.com/Shamrock-code/ShamrockWorkspace.git
```

Move in the created folder
```bash
cd ShamrockWorkspace
```
The workspace works in a similar way to Python venv, start by sourcing the `activate` script, it will register the PATH to utility script of the workspace and setup the Python Venv
```bash
source activate
```

On first run it will exit saying : 
```
> source activate   
-------------------------------------------
| Activate shamrock workspace environment |
-------------------------------------------

.
Shamrock workspace dir : ###SOME PATH###/ShamrockWorkspace


------ sycl compilers config ------
the compilers are not setup please run the following :
interactive_intelllvm_setup or interactive_acpp_setup
```
Because you have no SYCL compilers that are ready within the workspace.

## AdaptiveCpp setup

!!! info end "Installation instruction"
    To install recommanded package to compile AdaptiveCpp
    === "Linux (debian)"

        If you don't have llvm (...) : 

        ```bash
        wget --progress=bar:force https://apt.llvm.org/llvm.sh
        chmod +x llvm.sh
        sudo ./llvm.sh 16
        sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev
        sudo rm -r /usr/lib/clang/16*
        sudo ln -s /usr/lib/llvm-16/lib/clang/16 /usr/lib/clang/16
        ```

        for the other requirements :
        ```bash
        sudo apt install cmake libboost-all-dev python3-ipython
        ```

    === "MacOS"

        ```bash
        brew install cmake
        brew install libomp
        brew install boost
        ```

To setup AdaptativeCpp do :
```bash
interactive_acpp_setup
```

It should print : 
```
> interactive_acpp_setup
---------------------------------------
| AdaptiveCpp interactive setup       |
---------------------------------------

Shamrock workspace dir : ###SOME PATH###/ShamrockWorkspace
the submodule were not pulled
Cloning into 'AdaptiveCpp'...
remote: Enumerating objects: 22175, done.
remote: Counting objects: 100% (4481/4481), done.
remote: Compressing objects: 100% (659/659), done.
remote: Total 22175 (delta 4070), reused 3911 (delta 3813), pack-reused 17694
Receiving objects: 100% (22175/22175), 11.01 MiB | 3.84 MiB/s, done.
Resolving deltas: 100% (16349/16349), done.
use cuda [y/n]: n
```

Answer no (`n`) to `use cuda`, if you want to use GPUs you may want to read ... instead.
It will compile AdaptiveCpp and install it in a local folder within the workspace.


## Shamrock Download

You have to options now : 
 - Using your fork of shamrock
 - Using the main repo

If you want to use your fork clone it using (replace `<login>` by your github username):
```bash
git clone --recurse-submodules git@github.com:<login>/Shamrock.git
```
Otherwise you you want to use the main repo don't do anything.

Since you have now a SYCL compiler from the previsou step you can activate the workspace, it will create the python Venv and register the corect environment variables.
```bash
source activate
```

You see something like :
```
-------------------------------------------
| Activate shamrock workspace environment |
-------------------------------------------

.
Shamrock workspace dir : /home/tdavidcl/Documents/clean_test/ShamrockWorkspace


------ sycl compilers config ------
Intel llvm is not configured, to configure it run : interactive_intelllvm_setup
AdaptiveCpp dir : /home/tdavidcl/Documents/clean_test/ShamrockWorkspace/sycl_compilers/acpp/
---------------------------------------

------ python venv configuration ------
creating python venv :
Python venv dir  : /home/tdavidcl/Documents/clean_test/ShamrockWorkspace/ShamrockPyVenv
activating venv :
---------------------------------------

-> Succes
```
Maybe with some additional lines if shamrock was not pulled yet.

## Shamrock configuration

Just run the command :
```bash
configure_shamrock
```
The following script is provided by the ShamrockWorkspace.
It will configure all the possibles build configuration for the code.


## Finally: compile the code!

Move into the build directory and compile the code : 

```bash
cd Shamrock/build_config/acpp_omp_release/
make -j 4
```

Here we compile with only 4 process by default since the compiler can take up to 1 Gb per instance. If you have enough ram you can increase the number, or remove it to use the maximum number of threads.


## Remarks

If you want to setup the code with GPUs or do any advanced configuration please heads to those guides : ...