import os
import argparse

from lib.buildbot import * 




parser = argparse.ArgumentParser(description='Configure utility for the code')

parser.add_argument("--gen", help="use NINJA build system instead of Make")
parser.add_argument("--build", help="change the build type")



parser.add_argument("--nocode", action='store_true', help="disable the build of the main executable")
parser.add_argument("--lib", action='store_true', help="build the lib instead of the executable")
parser.add_argument("--tests", action='store_true', help="enable the build of the tests")

parser.add_argument("--outdir", help="output directory")


parser.add_argument("--cxxpath", action='store', help="select the compiler path")
parser.add_argument("--cxxcompiler", action='store', help="select the compiler")
parser.add_argument("--compiler", help="id of the compiler")
parser.add_argument("--profile", help="select the compilation profile")

parser.add_argument("--cxxflags", help="c++ compilation flags")
parser.add_argument("--cmakeargs", help="cmake configuration flags")


args = parser.parse_args()



abs_compiler_root_dir = ""

if args.cxxpath.startswith("/"):
    abs_compiler_root_dir = args.cxxpath
else:
    abs_compiler_root_dir = os.path.abspath(os.path.join(os.getcwd(),args.cxxpath))



print_buildbot_info("configure tool")



print("\033[1;34mCompiler directory \033[0;0m: "+ abs_compiler_root_dir)
print()


### interactive



### processing results

if args.outdir == None : 
    raise "no outdir specified"
    
print(args)





cmake_cmd = ""

cmake_cmd = "cmake"
cmake_cmd += " -S " + abs_src_dir + "/.."
cmake_cmd += " -B " + str(os.path.abspath(args.outdir))

### chose the generator
if args.gen == "ninja":
    cmake_cmd += " -G Ninja"
elif args.gen == "make":
    cmake_cmd += ' -G "Unix Makefiles"'
else:
    raise "unknown generator"





comp_path = ""

if args.cxxcompiler : 
    comp_path = args.cxxcompiler

### chose the compiler id
if args.compiler == "dpcpp":
    cmake_cmd += " -DSyCL_Compiler=DPCPP"
    if comp_path == "":
        comp_path = abs_compiler_root_dir + "/bin/clang++"

elif args.compiler == "opensycl":
    cmake_cmd += " -DSyCL_Compiler=OPENSYCL"
    if comp_path == "":
        comp_path = abs_compiler_root_dir + "/bin/syclcc"

else:
    cmake_cmd += " -DSyCL_Compiler=UNKNOWN"
    print("WARNING : The compiler is unknown")

    if not (args.profile == None):
        raise "can not select a profile with a unknown compiler"

    if (args.cxxcompiler == None):
        raise "you must select the compiler path if unknown"



cmake_cmd += " -DCMAKE_CXX_COMPILER="+comp_path


### pass if release or debug to cmake
if args.build == "release":
    cmake_cmd += " -DCMAKE_BUILD_TYPE=Release"
elif args.build == "debug":
    cmake_cmd += " -DCMAKE_BUILD_TYPE=Debug"



cmake_cmd += " -DCOMP_ROOT_DIR="+abs_compiler_root_dir


### target build
if (not args.nocode):
    cmake_cmd += " -DBUILD_SIM=true"

if (args.lib):
    cmake_cmd += " -DBUILD_PYLIB=true"

if (args.tests):
    cmake_cmd += " -DBUILD_TEST=true"








cpp_flags = ""

hipsyclconfigfile = "--hipsycl-config-file="+abs_compiler_root_dir+"/etc/hipSYCL/syclcc.json"


#-Xsycl-target-backend --cuda-gpu-arch=sm_86

profile_map = {
    "dpcpp" : {
        "bare" : "-fsycl",
        "cuda" : "-fsycl -fsycl-targets=nvptx64-nvidia-cuda",
        "cuda_sm86" : "-fsycl -fsycl-targets=nvidia_gpu_sm_86",
        "cuda_sm80" : "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80",
        "cuda_sm75" : "-fsycl -fsycl-targets=nvidia_gpu_sm_75",
        "cuda_sm72" : "-fsycl -fsycl-targets=nvidia_gpu_sm_72",
        "cuda_sm70" : "-fsycl -fsycl-targets=nvidia_gpu_sm_70",
        "cuda_sm62" : "-fsycl -fsycl-targets=nvidia_gpu_sm_62",
        "cuda_sm61" : "-fsycl -fsycl-targets=nvidia_gpu_sm_61",
        "cuda_sm60" : "-fsycl -fsycl-targets=nvidia_gpu_sm_60",
        "cuda_sm53" : "-fsycl -fsycl-targets=nvidia_gpu_sm_53",
        "cuda-profiling" : "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -g",
        "cuda-no-rdc" : "-fsycl -fno-sycl-rdc -fsycl-targets=nvptx64-nvidia-cuda",
        "cuda-index32bit" : "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-id-queries-fit-in-int",
        "hip-gfx906" : "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906",
    },
    "opensycl" : {
        "omp" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile,
        "omp_sanitizer" : "-fsanitize=address --hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile,
        "omp_coverage" : "-fprofile-instr-generate -fcoverage-mapping --hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile,
        "generic" : "--hipsycl-targets=generic "+ hipsyclconfigfile,
        "cuda-nvcxx" : "--hipsycl-targets='cuda-nvcxx' "+ hipsyclconfigfile,
        "cuda-sm70" : "--hipsycl-targets='cuda:sm_70' "+ hipsyclconfigfile,
        "hip-gfx906" : "--hipsycl-targets='hip:gfx906' "+ hipsyclconfigfile,

        #if you dare trying to develop with this profile
        "omp_insanity" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' -Wall -Wextra -Werror " + hipsyclconfigfile
    }
}

print(profile_map)

if not args.profile:
    print("WARNING : no profile were selected, you have to input the flag through 'cppflags'")
else:
    if not (args.profile in profile_map[args.compiler]):
        raise "the selected profile is unknown for this compiler"
    else:
        cpp_flags += profile_map[args.compiler][args.profile]



if args.cxxflags:
    if not (cpp_flags == ""):
        cpp_flags += " "
    cpp_flags += args.cxxflags

if args.cmakeargs:
    cmake_cmd += " " + args.cmakeargs.replace("\"","")




# TODO need a way to pass the configuration down aka is cuda or something else

if not (cpp_flags == ""):
    cmake_cmd += " -DCMAKE_CXX_FLAGS=\"" +cpp_flags+ "\""

run_cmd(cmake_cmd)