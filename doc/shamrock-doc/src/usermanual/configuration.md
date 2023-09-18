# Configuration

## Buildbot utility
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