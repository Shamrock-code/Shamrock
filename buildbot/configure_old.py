import os
import argparse

from lib.buildbot import *


parser = argparse.ArgumentParser(description='Configure utility for the code')

parser.add_argument("--ninja", action='store_true', help="use NINJA build system instead of Make")
parser.add_argument('--buildmode',action='store', type=str, default="None", help='target build mode')

parser.add_argument("--compiler", action='store_true', help="sycl compiler name")
parser.add_argument("--syclbe", action='store_true', help="Sycl backend to use")


parser.add_argument("--test", action='store_true', help="add test target to build configuration")
parser.add_argument("--shamrock", action='store_true', help="add shamrock target to build configuration")
parser.add_argument("--visu", action='store_true', help="add visualisation target to build configuration")

parser.add_argument('--interactive',     action='store_true', help='enables interactive configuration')

parser.add_argument("compiler_root",help="compiler location", type=str)

args = parser.parse_args()


print_buildbot_info("configure tool")

abs_build_dir = os.path.join(abs_proj_dir,"build")
abs_compiler_root_dir = os.path.abspath(os.path.join(os.getcwd(),args.compiler_root))

print("\033[1;34mCompiler directory \033[0;0m: "+ abs_compiler_root_dir)
print("\033[1;34mBuild directory    \033[0;0m: "+ abs_build_dir)
print()



if args.interactive:
    while 1:
        print("\033[1;34mInteractive configuration \033[0;0m:")

        args.ninja = input("    do you want to use ninja instead of make (y/n)") == "y"


        print("    compile mode available :")
        print("           0: Normal  (no special flags)")
        print("           1: Release (optimisation flags)")
        print("           2: Debug   (debug flags)")
        tmp_release = int(input("    with which mode do you want to compile :"))

        if tmp_release == 0:
            args.buildmode = "Normal"
        elif tmp_release == 1:
            args.buildmode = "Release"
        elif tmp_release == 2:
            args.buildmode = "Debug"

        args.shamrock = input("    do you want to compile the shamrock mode (y/n)") == "y"
        args.test = input("    do you want to compile the test mode (y/n)") == "y"
        args.visu = False

        args.compiler = input("    which compiler are you using (hipsycl/dpcpp)")

        args.backend = input("    which backend are you using (omp,cuda)")

        print("\033[1;34mOptions summary \033[0;0m: ")

        print("    ninja      =",args.ninja)
        print("    build mode =",args.buildmode)
        print("    compiler   =",args.compiler)
        print("    backend    =",args.backend)

        print("    shamrock   =",args.shamrock)
        print("    test       =",args.test)
        #print("    visu       =",args.visu)



        print()
        if input("confirm choices (y/N)") == "y":
            break
            print()
    print()


build_sys = BuildSystem.Makefiles
if args.ninja:
    build_sys = BuildSystem.Ninja




sycl_cmp = -1
if args.compiler == "dpcpp":
    sycl_cmp = SyclCompiler.DPCPP
    abs_build_dir += "_dpcpp"
elif args.compiler == "hipsycl":
    sycl_cmp = SyclCompiler.HipSYCL
    abs_build_dir += "_hipsycl"

sycl_be = -1
if args.backend == "omp":
    sycl_be = SyCLBE.OpenMP
elif args.backend == "cuda":
    sycl_be = SyCLBE.CUDA





target_buildmode = BuildMode.Normal
if args.buildmode == "Normal":
    target_buildmode = BuildMode.Normal

elif args.buildmode == "Release":
    target_buildmode = BuildMode.Release
    abs_build_dir += "_release"
elif args.buildmode == "Debug":
    target_buildmode = BuildMode.Debug
    abs_build_dir += "_debug"



target_lst = []
if args.shamrock:
    target_lst.append(Targets.SHAMROCK)
if args.test:
    target_lst.append(Targets.Test)
if args.visu:
    target_lst.append(Targets.Visu)










configure(
    abs_src_dir,
    abs_build_dir,
    sycl_cmp,
    sycl_be,
    abs_compiler_root_dir,
    target_buildmode,
    build_sys,
    target_lst)
