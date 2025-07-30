import argparse
import os

import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "UGA Kraken GPU H100 cluster"
PATH = "uga/kraken-gpu"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    args = parser.parse_args(argv)

    args.gen = "make"

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
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    envgen.gen_env_file("env_oneapi.sh")
