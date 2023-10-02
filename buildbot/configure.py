import os
import argparse

from lib.buildbot import * 

parser = argparse.ArgumentParser(description='Configure utility for Shamrock')

parser.add_argument("--compiler", help="select the id of the compiler config")
parser.add_argument("--cxxbin", action='store', help="override the executable chosen by --compiler")
parser.add_argument("--cxxpath", action='store', help="select the compiler root path")

parser.add_argument("--gen", help="generator to use (make, ninja, ...)")

parser.add_argument("--build", help="change the build type for shamrock")

parser.add_argument("--lib", action='store_true', help="build the lib instead of an executable (for python lib)")
parser.add_argument("--tests", action='store_true', help="build also the tests")

parser.add_argument("--builddir", help="build directory chosen")

parser.add_argument("--profile", help="select the compilation profile")

parser.add_argument("--cxxflags", help="additional c++ compilation flags")
parser.add_argument("--cmakeargs", help="additional cmake configuration flags")

parser.add_argument("--interactive",     action='store_true', help="additional cmake configuration flags")

args = parser.parse_args()


print_buildbot_info("configure tool")





if args.interactive:
    print("\033[1;34mInteractive configuration \033[0;0m:")
    args.cxxpath = input("    enter the compiler root dir path : ")
    print()


abs_compiler_root_dir = ""

if args.cxxpath.startswith("/"):
    abs_compiler_root_dir = args.cxxpath
else:
    abs_compiler_root_dir = os.path.abspath(os.path.join(os.getcwd(),args.cxxpath))







print("\033[1;34mCompiler directory \033[0;0m: "+ abs_compiler_root_dir)
print()




has_folder_hipsyclcmake = os.path.isdir(abs_compiler_root_dir + '/lib/cmake/hipSYCL')
has_folder_opensyclcmake = os.path.isdir(abs_compiler_root_dir + '/lib/cmake/OpenSYCL')
has_folder_acppcmake = os.path.isdir(abs_compiler_root_dir + '/lib/cmake/AdaptiveCpp')




if args.interactive:
    print("    possible compiler configs :")
    print("       1) intel_llvm")
    print("       2) AdaptiveCpp")
    #print("       3) AdaptiveCpp (cmake integration)")
    res = int(input("    select a compiler config (number of the config)"))

    if (res == 1):
        args.compiler = "intel_llvm"
    elif (res == 2):
        args.compiler = "acpp"
    elif (res == 3):
        args.compiler = "acpp_cmake"



comp_path = ""

if args.cxxbin : 
    comp_path = args.cxxbin

### chose the compiler id
if args.compiler == "intel_llvm":
    if comp_path == "":
        comp_path = abs_compiler_root_dir + "/bin/clang++"

elif args.compiler == "acpp":
    if comp_path == "":
        comp_path = abs_compiler_root_dir + "/bin/syclcc"

#elif args.compiler == "acpp_cmake":
#    if comp_path == "":
#        comp_path = ""

else:
    print("WARNING : The compiler is OTHER")

    if not (args.profile == None):
        raise "can not select a profile with a unknown compiler"

    if (args.cxxbin == None):
        raise "you must select the compiler path if unknown"





######### 
# defines all the build profiles
#########
intel_llvm_cmake_flag = "-DINTEL_LLVM_PATH="+abs_compiler_root_dir
profile_dpcpp = {
    "bare" : {"cxxflags" : "-fsycl", "cmakeflags" : intel_llvm_cmake_flag},
    "native_cpu" : {"cxxflags" : "-fsycl -fsycl-targets=native_cpu", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda" : {"cxxflags" : "-fsycl -fsycl-targets=nvptx64-nvidia-cuda", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm86" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_86", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm80" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_80", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm75" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_75", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm72" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_72", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm70" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_70", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm62" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_62", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm61" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_61", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm60" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_60", "cmakeflags" : intel_llvm_cmake_flag},
    "cuda_sm53" : {"cxxflags" : "-fsycl -fsycl-targets=nvidia_gpu_sm_53", "cmakeflags" : intel_llvm_cmake_flag},
    "hip-gfx906" : {"cxxflags" : "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906", "cmakeflags" : intel_llvm_cmake_flag},
}

hipsyclconfigfile = "--hipsycl-config-file="+abs_compiler_root_dir+"/etc/hipSYCL/syclcc.json"
acpp_path_cmake_flag = "-DACPP_PATH="+abs_compiler_root_dir
profile_acpp = {
    "omp" : {"cxxflags" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "omp_O3debug" : {"cxxflags" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' -g " + hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "omp_O3debugasan" : {"cxxflags" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' -g -fsanitize=address " + hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "omp_sanitizer" : {"cxxflags" : "-fsanitize=address --hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "omp_coverage" : {"cxxflags" : "-fprofile-instr-generate -fcoverage-mapping --hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "generic" : {"cxxflags" : "--hipsycl-targets=generic "+ hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "cuda-nvcxx" : {"cxxflags" : "--hipsycl-targets='cuda-nvcxx' "+ hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "cuda-sm70" : {"cxxflags" : "--hipsycl-targets='cuda:sm_70' "+ hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
    "hip-gfx906" : {"cxxflags" : "--hipsycl-targets='hip:gfx906' "+ hipsyclconfigfile, "cmakeflags" : acpp_path_cmake_flag},
}


hipsycl_cmake_dir = " -DhipSYCL_DIR="+abs_compiler_root_dir+"/lib/cmake/hipSYCL"
opensycl_cmake_dir = " -DOpenSYCL_DIR="+abs_compiler_root_dir+"/lib/cmake/OpenSYCL"
acpp_cmake_dir = " -DAdaptiveCpp_DIR="+abs_compiler_root_dir+"/lib/cmake/AdaptiveCpp"

cmake_dir_acpp_cmake = ""
acpp_cmake_name = ""
if(args.compiler == "acpp_cmake"):
    if(has_folder_acppcmake):
        cmake_dir_acpp_cmake = acpp_cmake_dir
        acpp_cmake_name = "ACPP"
    elif(has_folder_opensyclcmake):
        cmake_dir_acpp_cmake = opensycl_cmake_dir
        acpp_cmake_name = "OPENSYCL"
    elif(has_folder_hipsyclcmake):
        cmake_dir_acpp_cmake = hipsycl_cmake_dir
        acpp_cmake_name = "hipSYCL"
    else:
        raise "you want to use acpp_cmake mode but neither hipSYCL, OpenSYCL or AdaptiveCpp folders could be found in (cxxpath)/lib/cmake/"

profile_acpp_cmake = {
    "omp" : {"cxxflags" : "", "cmakeflags" : "-DUSE_ACPP_CMAKE=true -D"+acpp_cmake_name+"_TARGETS=omp "+cmake_dir_acpp_cmake},
    "generic" : {"cxxflags" : "", "cmakeflags" : "-DUSE_ACPP_CMAKE=true -D"+acpp_cmake_name+"_TARGETS=generic "+cmake_dir_acpp_cmake},
    "cuda-nvcxx" : {"cxxflags" : "", "cmakeflags" : "-DUSE_ACPP_CMAKE=true -D"+acpp_cmake_name+"_TARGETS=cuda-nvcxx "+cmake_dir_acpp_cmake},
    "cuda-sm70" : {"cxxflags" : "", "cmakeflags" : "-DUSE_ACPP_CMAKE=true -D"+acpp_cmake_name+"_TARGETS=cuda-sm70 "+cmake_dir_acpp_cmake},
    "hip-gfx906" : {"cxxflags" : "", "cmakeflags" : "-DUSE_ACPP_CMAKE=true -D"+acpp_cmake_name+"_TARGETS=hip-gfx906 "+cmake_dir_acpp_cmake},
}

#        "omp" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile,
#        "omp_O3debug" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' -g " + hipsyclconfigfile,
#        "omp_O3debugasan" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' -g -fsanitize=address " + hipsyclconfigfile,
#        "omp_sanitizer" : "-fsanitize=address --hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile,
#        "omp_coverage" : "-fprofile-instr-generate -fcoverage-mapping --hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' " + hipsyclconfigfile,
#        "generic" : "--hipsycl-targets=generic "+ hipsyclconfigfile,
#        "cuda-nvcxx" : "--hipsycl-targets='cuda-nvcxx' "+ hipsyclconfigfile,
#        "cuda-sm70" : "--hipsycl-targets='cuda:sm_70' "+ hipsyclconfigfile,
#        "hip-gfx906" : "--hipsycl-targets='hip:gfx906' "+ hipsyclconfigfile,
#
#        #if you dare trying to develop with this profile
#        "omp_insanity" : "--hipsycl-cpu-cxx=g++ --hipsycl-targets='omp' -Wall -Wextra -Werror " + hipsyclconfigfile
#






### interactive



if args.interactive:
    args.builddir = input("    please input the path to the build directory : ")


### processing results
if args.builddir == None : 
    raise "no output directory specified, please add --builddir flag pointing to the build folder"
    
#print(args)





cmake_cmd = ""

cmake_cmd = "cmake"
cmake_cmd += " -S " + abs_src_dir + "/.."
cmake_cmd += " -B " + str(os.path.abspath(args.builddir))



if args.interactive:
    print("    possible generator :")
    print("      1) ninja")
    print("      2) Makefiles")
    res = int(input("   input the generator to use : "))
    if (res == 1):
        args.gen = "ninja"
    elif (res == 2):
        args.gen = "make"


### chose the generator
if args.gen == "ninja":
    cmake_cmd += " -G Ninja"
elif args.gen == "make":
    cmake_cmd += ' -G "Unix Makefiles"'
else:
    raise "unknown generator"


if args.interactive:
    if args.compiler == "intel_llvm":
        print((profile_dpcpp.keys()))

    elif args.compiler == "acpp":
        print((profile_acpp.keys()))

    elif args.compiler == "acpp_cmake":
        print((profile_acpp_cmake.keys()))

    args.profile = input("    please input the profile name : ")



cpp_flags = ""
if not args.profile:
    print("WARNING : no profile were selected, you have to input the flag through 'cppflags'")
else:
    if args.compiler == "intel_llvm":
        cpp_flags += profile_dpcpp[args.profile]["cxxflags"]
        cmake_cmd += " "+profile_dpcpp[args.profile]["cmakeflags"]
        cmake_cmd += " -DCMAKE_CXX_COMPILER="+comp_path
        cmake_cmd += " -DSYCL_IMPLEMENTATION=IntelLLVM"

    elif args.compiler == "acpp":
        cpp_flags += profile_acpp[args.profile]["cxxflags"]
        cmake_cmd += " "+profile_acpp[args.profile]["cmakeflags"]
        cmake_cmd += " -DCMAKE_CXX_COMPILER="+comp_path
        cmake_cmd += " -DSYCL_IMPLEMENTATION=ACPPDirect"

    #elif args.compiler == "acpp_cmake":
    #    cpp_flags += profile_acpp_cmake[args.profile]["cxxflags"]
    #    cmake_cmd += " "+profile_acpp_cmake[args.profile]["cmakeflags"]




### pass if release or debug to cmake
if args.build == "release":
    cmake_cmd += " -DCMAKE_BUILD_TYPE=Release"
elif args.build == "debug":
    cmake_cmd += " -DCMAKE_BUILD_TYPE=Debug"
elif args.build == "coverage":
    cmake_cmd += " -DCMAKE_BUILD_TYPE=COVERAGE"
elif args.build == "asan":
    cmake_cmd += " -DCMAKE_BUILD_TYPE=ASAN"




cmake_cmd += " -DSHAMROCK_ENABLE_BACKEND=SYCL"


if (args.lib):
    cmake_cmd += " -DBUILD_PYLIB=true"

if (args.tests):
    cmake_cmd += " -DBUILD_TEST=true"














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
