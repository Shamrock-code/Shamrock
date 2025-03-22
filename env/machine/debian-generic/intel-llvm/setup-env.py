import argparse
import os

import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Debian generic Intel LLVM"
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

    parser.add_argument("--target", action="store", help="sycl backend to use")
    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    parser.add_argument("--cuda", action="store_true", help="build intel llvm with cuda support")
    parser.add_argument("--cuda-path", action="store", help="set cuda path")

    parser.add_argument("--hip", action="store_true", help="build intel llvm with hip support")
    parser.add_argument("--rocm-path", action="store", help="set ROCM path")

    args = parser.parse_args(argv)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    INTEL_LLVM_GIT_DIR = builddir + "/.env/intel-llvm-git"
    INTEL_LLVM_INSTALL_DIR = builddir + "/.env/intel-llvm-installdir"

    configure_args = utils.intel_llvm.get_llvm_configure_arg(args)
    shamcxx_args = utils.intel_llvm.get_intel_llvm_target_flags(args)

    ENV_SCRIPT_PATH = builddir + "/activate"

    ENV_SCRIPT_HEADER = ""
    ENV_SCRIPT_HEADER += "export SHAMROCK_DIR=" + shamrockdir + "\n"
    ENV_SCRIPT_HEADER += "export BUILD_DIR=" + builddir + "\n"
    ENV_SCRIPT_HEADER += "export INTEL_LLVM_GIT_DIR=" + INTEL_LLVM_GIT_DIR + "\n"
    ENV_SCRIPT_HEADER += "export INTEL_LLVM_INSTALL_DIR=" + INTEL_LLVM_INSTALL_DIR + "\n"
    ENV_SCRIPT_HEADER += "export INTELLLVM_CONFIGURE_ARGS=(" + configure_args + ")\n"
    ENV_SCRIPT_HEADER += 'export INTEL_LLVM_VERSION="nightly-2024-10-27"\n'

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

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    cmake_extra_args = ""
    if pylib:
        run_cmd(
            "cp "
            + os.path.abspath(os.path.join(cur_file, "../" + "_pysetup.py"))
            + " "
            + builddir
            + "/setup.py"
        )

    if lib_mode == "shared":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=On"
    elif lib_mode == "object":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=Off"

    ENV_SCRIPT_HEADER += "export CMAKE_OPT=(" + cmake_extra_args + ")\n"
    ENV_SCRIPT_HEADER += 'export SHAMROCK_CXX_FLAGS="' + shamcxx_args + '"\n'
    ENV_SCRIPT_HEADER += 'export SHAMROCK_BUILD_TYPE="' + cmake_build_type + '"\n'
    ENV_SCRIPT_HEADER += "\n"

    source_file = "env_built_intel-llvm.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../" + source_file))

    utils.envscript.write_env_file(
        source_path=source_path, header=ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
    )
