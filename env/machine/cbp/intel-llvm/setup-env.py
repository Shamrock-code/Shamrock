import argparse
import os

import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "CBP Machines - Intel LLVM"
PATH = "machine/cbp/intel-llvm"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    parser.add_argument(
        "--custommpi", action="store_true", help="build shamrock with custom mpi support"
    )
    parser.add_argument("--ucxurl", action="store", help="ucx source url")
    parser.add_argument("--ompiurl", action="store", help="ompi source url")

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
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/clone-intel-llvm.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    if args.custommpi:

        UCX_INSTALL_PATH = builddir + "/.env/ucx-install"
        OMPI_INSTALL_PATH = builddir + "/.env/ompi-install"
        OMPI_SOURCE_DIR = builddir + "/.env/ompi-sources"
        CUDA_PATH = "/usr/lib/cuda"

        if args.ucxurl is None or args.ompiurl is None:
            raise "ucxurl and ompiurl must be set if custom mpi on"

        envgen.ENV_SCRIPT_HEADER += "#### mpi setup ####\n"
        envgen.ENV_SCRIPT_HEADER += 'export UCX_URL="' + args.ucxurl + '"\n'
        envgen.ENV_SCRIPT_HEADER += 'export OMPI_URL="' + args.ompiurl + '"\n'
        envgen.ENV_SCRIPT_HEADER += 'export CUDA_PATH="' + CUDA_PATH + '"\n'

        envgen.ENV_SCRIPT_HEADER += 'export UCX_INSTALL_PATH="' + UCX_INSTALL_PATH + '"\n'
        envgen.ENV_SCRIPT_HEADER += 'export OMPI_INSTALL_PATH="' + OMPI_INSTALL_PATH + '"\n'
        envgen.ENV_SCRIPT_HEADER += 'export OMPI_SOURCE_DIR="' + OMPI_SOURCE_DIR + '"\n'
        envgen.ENV_SCRIPT_HEADER += "#### ######### ####\n"

        envgen.ext_script_list.append(shamrockdir + "/env/helpers/setup_mpi.sh")

    envgen.gen_env_file("env_built_intel-llvm.sh")
    envgen.copy_env_file("binding_script.sh", "binding_script.sh")
