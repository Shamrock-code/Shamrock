import argparse
import os

import utils.acpp
import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.sysinfo
from utils.setuparg import *

NAME = "OzStar Ngarrgu Tindebeek (AdaptiveCpp) A100"
PATH = "machine/ozstar/ngarrgu-tindebeek"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    args = parser.parse_args(argv)
    args.gen = "ninja"

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
        shamrockdir + "/env/helpers/clone-acpp.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    envgen.gen_env_file("env_built_acpp.sh")
    # envgen.copy_env_file("binding_script.sh", "binding_script.sh")
