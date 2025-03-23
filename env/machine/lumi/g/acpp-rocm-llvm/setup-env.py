import argparse
import os

import utils.amd_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Lumi-G Intel AdaptiveCpp ROCM/LLVM"
PATH = "machine/lumi/standard-g/acpp-rocm-llvm"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    pylib = arg.pylib
    lib_mode = arg.lib_mode

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    if pylib:
        print("this env does not support --pylib")
        raise ""

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    args = parser.parse_args(argv)
    args.gen = "ninja"

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    ##############################
    # Generate env script header
    ##############################

    cmake_extra_args = ""
    envgen.export_list = {
        "SHAMROCK_DIR": shamrockdir,
        "BUILD_DIR": builddir,
        "CMAKE_GENERATOR": cmake_gen,
        "MAKE_EXEC": gen,
        "MAKE_OPT": f"({gen_opt})",
        "CMAKE_OPT": f"({cmake_extra_args})",
        "SHAMROCK_BUILD_TYPE": f"'{cmake_build_type}'",
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/clone-acpp.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    envgen.copy_env_file("exemple_batch.sh", "exemple_batch.sh")
    envgen.gen_env_file("env_built_acpp.sh")
