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


def is_intel_llvm_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/clang++")


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

    run_cmd("mkdir -p " + builddir)
    run_cmd("mkdir -p " + builddir + "/.env")

    ENV_SCRIPT_PATH = builddir + "/activate"
    ENV_SCRIPT_HEADER = ""

    ##############################
    # Generate env script header
    ##############################

    cmake_extra_args = ""
    export_list = {
        "SHAMROCK_DIR": shamrockdir,
        "BUILD_DIR": builddir,
        "CMAKE_GENERATOR": cmake_gen,
        "MAKE_EXEC": gen,
        "MAKE_OPT": f"({gen_opt})",
        "CMAKE_OPT": f"({cmake_extra_args})",
        "SHAMROCK_BUILD_TYPE": f"'{cmake_build_type}'",
    }

    ext_script_list = [
        shamrockdir + "/env/helpers/clone-acpp.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    for k in export_list.keys():
        ENV_SCRIPT_HEADER += "export " + k + "=" + export_list[k] + "\n"

    spacer = "\n####################################################################################################"

    for f in ext_script_list:
        ENV_SCRIPT_HEADER += f"{spacer}\n# Imported script " + f + f"{spacer}\n\n"
        ENV_SCRIPT_HEADER += utils.envscript.file_to_string(f)
        ENV_SCRIPT_HEADER += f"{spacer}{spacer}{spacer}\n"

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
