# Configuration

## Buildbot utility
```
usage: configure.py [-h] [--compiler COMPILER] [--cxxbin CXXBIN] [--cxxpath CXXPATH] [--gen GEN] [--build BUILD] [--lib] [--tests]
                    [--builddir BUILDDIR] [--profile PROFILE] [--cxxflags CXXFLAGS] [--cmakeargs CMAKEARGS]

Configure utility for Shamrock

options:
  -h, --help            show this help message and exit
  --compiler COMPILER   select the id of the compiler config
  --cxxbin CXXBIN       override the executable chosen by --compiler
  --cxxpath CXXPATH     select the compiler root path
  --gen GEN             generator to use (make, ninja, ...)
  --build BUILD         change the build type for shamrock
  --lib                 build the lib instead of an executable (for python lib)
  --tests               build also the tests
  --builddir BUILDDIR   build directory chosen
  --profile PROFILE     select the compilation profile
  --cxxflags CXXFLAGS   additional c++ compilation flags
  --cmakeargs CMAKEARGS
                        additional cmake configuration flags
```

Essentially for each compiler we define several profiles, which holds corresponding compilation flags.
One exemple could be :
```bash
python buildbot/configure.py
    --gen ninja                     #to compile with ninja instead of make
    --build debug                   #to compile the code in debug mode
    --tests                         #compile also the test
    --builddir build_hipsycl_debug    #directory where the build files will be held
    --cxxpath ../sycl_cpl/hipSYCL   #path to the cxx compiler
    --compiler hipsycl              #id of the compiler (hipsycl or dpcpp)
    --profile omp                   #selected profile
```

After this command ran, you can `cd` into the build directory and type `ninja` or `make -J` depending on the build system selected.


## Installing the code as a python module

By running this command you can pip install the code into you local python installation, and open shamrock in your prefered environment.
```bash
BUILDBOT_ARGS="--cxxpath <PATH TO COMPILER> --compiler <COMPILER id> --profile <PROFILE>" pip install .
```
