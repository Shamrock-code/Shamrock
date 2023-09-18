# Getting Started


## Download and compile 

To get started since we use SYCL we use specific compilers that can compile SYCL c++ code to any SYCL backend. 
If you already have a SYCL compiler skip this step and move to [Compile shamrock](#compile-shamrock) section.

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

Clone the OpenSYCL repository: 
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

if your
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

now move out of OpenSYCL direcotry
```sh
cd ..
```
, you should see a `OpenSYCL_comp` folder.

## Compile Shamrock

First go on [Shamrock repo](https://github.com/tdavidcl/Shamrock) and fork the code. You can now clone your fork on your laptop, desktop, ... : (replace `github_username` by your github id)

```bash
git clone --recurse-submodules git@github.com:github_username/Shamrock.git
```

move in the Shamrock folder

```sh
cd Shamrock
```

For configuration since cmake arguments can become quite complex 
I wrote a configuration utility to avoid dealing with that madness 

```
python3 buildbot/configure.py \
  --gen make \
  --build release \
  --tests \
  --outdir build \
  --cxxpath ../OpenSYCL_comp \
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


