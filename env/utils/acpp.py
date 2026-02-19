import os

import utils.amd_arch
import utils.cuda_arch
from utils.oscmd import *


def clone_acpp(folder):
    if os.path.isdir(folder):
        print("-- skipping git clone folder does already exist")
    else:
        print("-- clonning https://github.com/AdaptiveCpp/AdaptiveCpp.git")
        run_cmd("git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git " + folder)


def get_acpp_target_env(args, version="git"):
    backend = "omp"

    backend_list = [
        "omp",
        "omp.accelerated",
        "omp.library-only",
        "generic",
        "cuda",
        "cuda.explicit-multipass",
        "cuda.integrated-multipass",
        "hip",
        "hip.integrated-multipass",
    ]

    if not (args.backend == None):
        if args.backend in ["omp", "omp.accelerated", "omp.library-only", "generic"]:
            return args.backend

        elif args.backend in ["cuda", "cuda.explicit-multipass", "cuda.integrated-multipass"]:
            utils.cuda_arch.print_description(args.arch)

            return args.backend + ":" + args.arch

        elif args.backend in ["hip", "hip.integrated-multipass"]:
            utils.amd_arch.print_description(args.arch)

            return args.backend + ":" + args.arch

        else:
            print("-- you must chose a backend in the following list:")
            for b in backend_list:
                print("     ", b)
            raise ValueError(f"unknown acpp backend: {args.backend}")
