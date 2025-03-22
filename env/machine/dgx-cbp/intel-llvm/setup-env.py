import argparse
import os

import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "CBP Nvidia DGX A100 Intel LLVM CUDA"
PATH = "machine/debian-generic/intel-llvm"


def is_intel_llvm_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/clang++")


def setup(arg: SetupArg):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    pylib = arg.pylib
    lib_mode = arg.lib_mode

    print("------------------------------------------")
    print("Running env setup for : " + NAME)
    print("------------------------------------------")

    if pylib:
        print("this env does not support --pylib")
        raise ""

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    parser.add_argument(
        "--custommpi", action="store_true", help="build shamrock with custom mpi support"
    )
    parser.add_argument("--ucxurl", action="store", help="ucx source url")
    parser.add_argument("--ompiurl", action="store", help="ompi source url")

    args = parser.parse_args(argv)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    INTEL_LLVM_GIT_DIR = builddir + "/.env/intel-llvm-git"
    INTEL_LLVM_INSTALL_DIR = builddir + "/.env/intel-llvm-installdir"

    ENV_SCRIPT_PATH = builddir + "/activate"

    ENV_SCRIPT_HEADER = ""
    ENV_SCRIPT_HEADER += "export SHAMROCK_DIR=" + shamrockdir + "\n"
    ENV_SCRIPT_HEADER += "export BUILD_DIR=" + builddir + "\n"
    ENV_SCRIPT_HEADER += "export INTEL_LLVM_GIT_DIR=" + INTEL_LLVM_GIT_DIR + "\n"
    ENV_SCRIPT_HEADER += "export INTEL_LLVM_INSTALL_DIR=" + INTEL_LLVM_INSTALL_DIR + "\n"

    run_cmd("mkdir -p " + builddir + "/.env")

    INTEL_LLVM_CLONE_HELPER = builddir + "/.env/clone-llvm"
    utils.envscript.write_env_file(
        source_path=shamrockdir + "/env/helpers/clone-intel-llvm.sh",
        header="",
        path_write=INTEL_LLVM_CLONE_HELPER,
    )
    ENV_SCRIPT_HEADER += ". " + INTEL_LLVM_CLONE_HELPER + "\n"

    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += 'export CMAKE_GENERATOR="' + cmake_gen + '"\n'
    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export MAKE_EXEC=" + gen + "\n"
    ENV_SCRIPT_HEADER += "export MAKE_OPT=(" + gen_opt + ")\n"
    cmake_extra_args = ""
    ENV_SCRIPT_HEADER += "export CMAKE_OPT=(" + cmake_extra_args + ")\n"
    ENV_SCRIPT_HEADER += 'export SHAMROCK_BUILD_TYPE="' + cmake_build_type + '"\n'
    ENV_SCRIPT_HEADER += "\n"

    if args.custommpi:

        UCX_INSTALL_PATH = builddir + "/.env/ucx-install"
        OMPI_INSTALL_PATH = builddir + "/.env/ompi-install"
        OMPI_SOURCE_DIR = builddir + "/.env/ompi-sources"
        CUDA_PATH = "/usr/lib/cuda"

        MPI_ENV = builddir + "/.env/activate-ompi"

        if args.ucxurl is None or args.ompiurl is None:
            raise "ucxurl and ompiurl must be set if custom mpi on"

        OMPI_HEADER = ""
        ENV_SCRIPT_HEADER += "#### mpi setup ####\n"
        ENV_SCRIPT_HEADER += 'export UCX_URL="' + args.ucxurl + '"\n'
        ENV_SCRIPT_HEADER += 'export OMPI_URL="' + args.ompiurl + '"\n'
        ENV_SCRIPT_HEADER += 'export CUDA_PATH="' + CUDA_PATH + '"\n'

        ENV_SCRIPT_HEADER += 'export UCX_INSTALL_PATH="' + UCX_INSTALL_PATH + '"\n'
        ENV_SCRIPT_HEADER += 'export OMPI_INSTALL_PATH="' + OMPI_INSTALL_PATH + '"\n'
        ENV_SCRIPT_HEADER += 'export OMPI_SOURCE_DIR="' + OMPI_SOURCE_DIR + '"\n'
        ENV_SCRIPT_HEADER += ". " + MPI_ENV + "\n"
        ENV_SCRIPT_HEADER += "#### ######### ####\n"

        utils.envscript.write_env_file(
            source_path=shamrockdir + "/env/helpers/setup_mpi.sh", header="", path_write=MPI_ENV
        )

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))
    source_file = "env_built_intel-llvm.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../" + source_file))

    utils.envscript.write_env_file(
        source_path=source_path, header=ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
    )

    if is_intel_llvm_already_installed(INTEL_LLVM_INSTALL_DIR):
        print("-- intel llvm already installed => skipping")
    else:
        print("-- running compiler setup")
        os.system("bash -c 'cd " + builddir + " && source ./activate &&  updatecompiler'")
