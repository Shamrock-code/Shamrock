import argparse
import os

import utils.amd_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Lumi-G Intel AdaptiveCpp ROCM"
PATH = "machine/lumi/standard-g/acpp"


def is_intel_llvm_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/clang++")


def setup(arg: SetupArg):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    pylib = arg.pylib
    lib_mode = arg.lib_mode

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    print("------------------------------------------")
    print("Running env setup for : " + NAME)
    print("------------------------------------------")

    if pylib:
        print("this env does not support --pylib")
        raise ""

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    args = parser.parse_args(argv)
    args.gen = "ninja"

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    run_cmd("mkdir -p " + builddir)
    run_cmd("mkdir -p " + builddir + "/.env")

    ACPP_GIT_DIR = builddir + "/.env/acpp-git"
    ACPP_BUILD_DIR = builddir + "/.env/acpp-builddir"
    ACPP_INSTALL_DIR = builddir + "/.env/acpp-installdir"

    ENV_SCRIPT_PATH = builddir + "/activate"

    ##############################
    # Generate env script header
    ##############################
    ENV_SCRIPT_HEADER = ""
    ENV_SCRIPT_HEADER += "export SHAMROCK_DIR=" + shamrockdir + "\n"
    ENV_SCRIPT_HEADER += "export BUILD_DIR=" + builddir + "\n"
    ENV_SCRIPT_HEADER += "export ACPP_GIT_DIR=" + ACPP_GIT_DIR + "\n"
    ENV_SCRIPT_HEADER += "export ACPP_BUILD_DIR=" + ACPP_BUILD_DIR + "\n"
    ENV_SCRIPT_HEADER += "export ACPP_INSTALL_DIR=" + ACPP_INSTALL_DIR + "\n"

    ACPP_CLONE_HELPER = builddir + "/.env/clone-acpp"
    utils.envscript.write_env_file(
        source_path=shamrockdir + "/env/helpers/clone-acpp.sh",
        header="",
        path_write=ACPP_CLONE_HELPER,
    )

    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += 'export CMAKE_GENERATOR="' + cmake_gen + '"\n'
    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export MAKE_EXEC=" + gen + "\n"
    ENV_SCRIPT_HEADER += "export MAKE_OPT=(" + gen_opt + ")\n"
    cmake_extra_args = ""
    ENV_SCRIPT_HEADER += "export CMAKE_OPT=(" + cmake_extra_args + ")\n"
    ENV_SCRIPT_HEADER += 'export SHAMROCK_BUILD_TYPE="' + cmake_build_type + '"\n'
    ENV_SCRIPT_HEADER += "\n"

    exemple_batch_file = "exemple_batch.sh"
    exemple_batch_path = os.path.abspath(os.path.join(cur_file, "../" + exemple_batch_file))
    utils.envscript.copy_env_file(
        source_path=exemple_batch_path, path_write=builddir + "/exemple_batch.sh"
    )

    source_file = "env_built_acpp.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../" + source_file))

    utils.envscript.write_env_file(
        source_path=source_path, header=ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
    )
