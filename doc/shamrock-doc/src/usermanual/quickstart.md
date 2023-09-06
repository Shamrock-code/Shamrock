# Getting Started


## Download and compile 

First make a directory to work in and move in it: 

```bash
mkdir ShamrockWorkspace
cd ShamrockWorkspace
```
### Setup the compiler (openSYCL)

Install requirements : 
<table>
<tr>
<th>Linux (debian)</th>
<th>MacOS</th>
</tr>
<tr>
<td valign="top">
Install llvm and other requirements

```bash
wget --progress=bar:force https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 16
sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev
sudo rm -r /usr/lib/clang/16*
sudo ln -s /usr/lib/llvm-16/lib/clang/16 /usr/lib/clang/16

sudo apt install cmake libboost-all-dev
```
</td>
<td valign="top">

```bash
brew install cmake
brew install libomp
brew install boost
```
</td>
</tr>
</table>

Dowload OpenSYCL : 

```bash
git clone --recurse-submodules https://github.com/OpenSYCL/OpenSYCL.git
cd OpenSYCL
```

Configure OpenSYCL : 

<table>
<tr>
<th>Linux (debian)</th>
<th>MacOS</th>
</tr>
<tr>
<td valign="top">

```bash
cmake \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-16 \
    -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-16 \
    -DCMAKE_INSTALL_PREFIX=../OpenSYCL_comp .
```
</td>
<td valign="top">

```bash
OMP_ROOT=` brew list libomp | 
    grep libomp.a | 
    sed -E "s/\/lib\/.*//"`

cmake \
    -DOpenMP_ROOT=$OMP_ROOT \
    -DCMAKE_INSTALL_PREFIX=../OpenSYCL_comp .
```
</td>
</tr>
</table>

Compile OpenSYCL : 

```bash
make -j install
```

now move out of the main directory, you should see a `OpenSYCL_comp` folder

## Compile Shamrock

First go on [Shamrock repo](https://github.com/tdavidcl/Shamrock) and fork the code. You can now clone your fork on your laptop, desktop, ... : (replace `github_username` by your github id)

```bash
git clone --recurse-submodules git@github.com:github_username/Shamrock.git
cd Shamrock
```

For configuration since cmake arguments can become quite complex 
I wrote a configuration utility to avoid dealing with that madness 

```
python3 buildbot/configure.py 
  --gen make 
  --build release 
  --tests 
  --outdir build 
  --cxxpath ../OpenSYCL_comp 
  --compiler opensycl
```

Here we tell the configure utility :
- `--gen make`, to use `make` for project generation (alternatively you can use `ninja` if it is installed it may reduce the build time significately). 
- `--build release` tells the utility to compile an optimized version. 
- `--outdir build` mean that the build directory will be `build`
- `--cxxpath ../OpenSYCL_comp` tells the path to the used compiler (opensycl here)
- `--compiler opensycl` tells the code that OpenSYCL is used.

Move into the build directory and compile the code : 

```bash
cd build
make -j 4
```

Here we compile with only 4 process by default since the compiler can take up to 1 Gb per instance. If you have enough ram you can increse the number, or remove it to use the maximum number of threads.







## Old tuto



## Supported backends



| Support Matrix |                 |                |         
|--------------|-----------------|----------------|
|              | DPCPP           | HipSYCL        |         
|--------------|-----------------|----------------|
| Host         | V               |      X         | 
| OpenMP       | X               |       V        |      
| CUDA         |       V         |       V        |  

## SYCL Setup

```bash
wget https://raw.githubusercontent.com/tdavidcl/sycl-setup-script/main/setup_sycl.sh
sh setup_sycl.sh
```

if you plan on using dpcpp run in terminal : 
```bash
export DPCPP_HOME=$(pwd)/sycl_cpl/dpcpp 
export LD_LIBRARY_PATH=$DPCPP_HOME/lib:$LD_LIBRARY_PATH
export PATH=$DPCPP_HOME/bin:$PATH
```


## Configuration & Compilation

Download the repo : 
```bash
git clone git@github.com:tdavidcl/Shamrock.git
```

We provide a configuration utility to help dealing with the wide variety of configuration to compile sycl code.
```bash
python buildbot/configure.py
```

```
usage: configure.py [-h] [--gen GEN] [--build BUILD] [--nocode] [--lib] [--tests] [--outdir OUTDIR] [--cxxpath CXXPATH] [--cxxcompiler CXXCOMPILER] [--compiler COMPILER] [--profile PROFILE]
                    [--cppflags CPPFLAGS] [--cmakeargs CMAKEARGS]

Configure utility for the code

options:
  -h, --help            show this help message and exit
  --gen GEN             use NINJA build system instead of Make
  --build BUILD         change the build type
  --nocode              disable the build of the main executable
  --lib                 build the lib instead of the executable
  --tests               enable the build of the tests
  --outdir OUTDIR       output directory
  --cxxpath CXXPATH     select the compiler path
  --cxxcompiler CXXCOMPILER
                        select the compiler
  --compiler COMPILER   id of the compiler
  --profile PROFILE     select the compilation profile
  --cppflags CPPFLAGS   c++ compilation flags
  --cmakeargs CMAKEARGS
                        cmake configuration flags

```

Essentially for each compiler we define several profiles, which holds corresponding compilation flags. 
One exemple could be :
```bash
python buildbot/configure.py 
    --gen ninja                     #to compile with ninja instead of make
    --build debug                   #to compile the code in debug mode
    --tests                         #compile also the test
    --outdir build_hipsycl_debug    #directory where the build files will be held
    --cxxpath ../sycl_cpl/hipSYCL   #path to the cxx compiler
    --compiler hipsycl              #id of the compiler (hipsycl or dpcpp)
    --profile omp                   #selected profile
```

After this command ran, you can `cd` into the build directory and type `ninja` or `make -J` depending on the build system selected.


## Installing the code as a python module

By running this command you can pip install the code into you local python installation, and open shamrock in your prefered environment.
```bash
BUILDBOT_ARGS="--cxxpath <PATH TO COMPILER> --compiler <COMPILER id> --profile <PROFILE>" pip install -e .
```