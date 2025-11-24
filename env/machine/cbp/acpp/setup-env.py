import argparse
import os
import subprocess

import utils.acpp
import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.sysinfo
from utils.setuparg import *

NAME = "CBP Machines - AdaptiveCpp"
PATH = "machine/cbp/acpp"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    def has_cmd(cmd: str) -> bool:
        return (
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
            ).returncode
            == 0
        )

    if has_cmd("nvidia-smi"):
        default_backend = "cuda"
    elif has_cmd("rocm-smi"):
        default_backend = "rocm"
    else:
        default_backend = "x86"
    parser.add_argument(
        "--backend",
        type=str,
        action="store",
        help="backend to build ACPP with (cuda, sscp, rocm or x86)",
        default=default_backend,
    )

    args = parser.parse_args(argv)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    cmake_extra_args = ""

    envgen.export_list = {
        "SHAMROCK_DIR": shamrockdir,
        "BUILD_DIR": builddir,
        "CMAKE_GENERATOR": cmake_gen,
        "MAKE_EXEC": gen,
        "MAKE_OPT": f"({gen_opt})",
        "CMAKE_OPT": f"({cmake_extra_args})",
        "SHAMROCK_BUILD_TYPE": f"'{cmake_build_type}'",
        "ACPP_BACKEND": f"{args.backend}",
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/clone-acpp.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
        shamrockdir + "/env/helpers/clone-llvm.sh",
    ]

    envgen.gen_env_file("env_built_acpp.sh")
    envgen.copy_env_file("binding_script.sh", "binding_script.sh")
