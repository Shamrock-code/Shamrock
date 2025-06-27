import os

import utils.amd_arch
import utils.cuda_arch
from utils.oscmd import *


def clone_intel_llvm(folder):
    if os.path.isdir(folder):
        print("-- skipping git clone folder does already exist")
    else:
        print("-- clonning https://github.com/intel/llvm.git")
        run_cmd("git clone https://github.com/intel/llvm.git " + folder)


def change_branch(folder, branch_name):
    run_cmd("cd " + folder + " && git checkout " + branch_name)


def get_llvm_configure_arg(args):
    configure_args = ""
    if args.cuda:
        configure_args += " --cuda"
    if not (args.cuda_path == None):
        pth = os.path.abspath(os.path.expanduser(args.cuda_path))
        configure_args += ' --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=' + pth + '"'

    if args.hip:
        configure_args += " --hip"
    if not (args.rocm_path == None):
        pth = os.path.abspath(os.path.expanduser(args.rocm_path))
        configure_args += ' --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=' + pth + '"'

    return configure_args


def get_intel_llvm_target_flags(args):
    if args.target == None:
        raise Exception('You must chose a target (using the "--target" flag)')

    arch_list = {}

    if args.cuda:
        for k in utils.cuda_arch.NVIDIA_ARCH_DESC.keys():
            arch_list["nvidia_gpu_" + k] = {
                "desc": utils.cuda_arch.NVIDIA_ARCH_DESC[k],
                "flags": "-fsycl -fsycl-targets=nvidia_gpu_" + k,
            }

    if args.hip:
        for k in utils.amd_arch.AMD_ARCH_DESC.keys():
            arch_list["amd_gpu_" + k] = {
                "desc": utils.amd_arch.AMD_ARCH_DESC[k],
                "flags": "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch="
                + k,
            }

    if (args.cuda == args.hip) and (args.cuda == False):
        raise Exception(
            'No backend enabled for intel llvm (use the "--cuda" or "--hip" flag to enable one)'
        )

    if not (args.target in arch_list.keys()):
        print(f"enabled target (Cuda={args.cuda}, Hip={args.hip}) :")
        for k in arch_list.keys():
            print("   ", k, ":", arch_list[k]["desc"])
        raise Exception("Unknown target")

    print("you've chosen intel llvm backend :", args.target)
    print("     ->", arch_list[args.target]["desc"])

    return arch_list[args.target]["flags"]
