import argparse
import os

import utils.acpp
import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Debian generic AdaptiveCpp"
PATH = "machine/debian-generic/acpp"


def is_acpp_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/acpp")


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

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--backend", action="store", help="sycl backend to use")
    parser.add_argument("--arch", action="store", help="arch to build")
    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    args = parser.parse_args(argv)

    acpp_target = utils.acpp.get_acpp_target_env(args)
    if acpp_target == None:
        print("-- target not specified using acpp default")
    else:
        print("-- setting acpp target to :", acpp_target)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    ENV_SCRIPT_PATH = builddir + "/activate"

    run_cmd("mkdir -p " + builddir + "/.env")

    ENV_SCRIPT_HEADER = ""

    cmake_extra_args = ""
    if lib_mode == "shared":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=On"
    elif lib_mode == "object":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=Off"

    ACPP_GIT_DIR = builddir + "/.env/acpp-git"
    ACPP_BUILD_DIR = builddir + "/.env/acpp-builddir"
    ACPP_INSTALL_DIR = builddir + "/.env/acpp-installdir"

    export_list = {
        "SHAMROCK_DIR" : shamrockdir,
        "BUILD_DIR" : builddir,
        "ACPP_GIT_DIR": ACPP_GIT_DIR,
        "ACPP_BUILD_DIR": ACPP_BUILD_DIR,
        "ACPP_INSTALL_DIR": ACPP_INSTALL_DIR,
        "CMAKE_GENERATOR": cmake_gen,
        "MAKE_EXEC" : gen,
        "MAKE_OPT" : f"({gen_opt})",
        "CMAKE_OPT" : f"({cmake_extra_args})",
        "SHAMROCK_BUILD_TYPE" : f"'{cmake_build_type}'",
        "SHAMROCK_CXX_FLAGS" : "\" --acpp-targets='" + acpp_target + "'\""
    }

    ext_script_list = [
       shamrockdir + "/env/helpers/clone-acpp.sh",
       shamrockdir + "/env/helpers/pull_reffiles.sh"
    ]
    
    for k in export_list.keys():
        ENV_SCRIPT_HEADER += "export " + k + "=" + export_list[k] + "\n"

    spacer ="\n####################################################################################"
    for f in ext_script_list:
        ENV_SCRIPT_HEADER += f"{spacer}\n# Imported script " + f + f"{spacer}\n"
        ENV_SCRIPT_HEADER += utils.envscript.file_to_string(f)
        ENV_SCRIPT_HEADER += f"{spacer}{spacer}{spacer}\n"

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    source_file = "env_built_acpp.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../" + source_file))

    utils.envscript.write_env_file(
        source_path=source_path, header=ENV_SCRIPT_HEADER, path_write=ENV_SCRIPT_PATH
    )

    if pylib:
        run_cmd(
            "cp "
            + os.path.abspath(os.path.join(cur_file, "../" + "_pysetup.py"))
            + " "
            + builddir
            + "/setup.py"
        )
