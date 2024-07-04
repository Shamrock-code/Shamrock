import argparse
import os
import utils.acpp
import utils.sysinfo
import utils.envscript
import utils.cuda_arch
import utils.amd_arch

NAME = "Debian generic AdaptiveCpp"
PATH = "machine/debian-generic/acpp"

def is_acpp_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/acpp")

def setup(argv,builddir, shamrockdir,buildtype):

    print("------------------------------------------")
    print("Running env setup for : "+NAME)
    print("------------------------------------------")

    parser = argparse.ArgumentParser(prog=PATH,description= NAME+' env for Shamrock')

    parser.add_argument("--backend", action='store', help="sycl backend to use")
    parser.add_argument("--arch", action='store', help="arch to build")
    parser.add_argument("--gen", action='store', help="generator to use (ninja or make)")

    args = parser.parse_args(argv)

    acpp_target = utils.acpp.get_acpp_target_env(args)  
    if (acpp_target == None):
        print("-- target not specified using acpp default")
    else:
        print("-- setting acpp target to :",acpp_target)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)
    
    ACPP_GIT_DIR = builddir+"/.env/acpp-git"
    ACPP_BUILD_DIR = builddir + "/.env/acpp-builddir"
    ACPP_INSTALL_DIR = builddir + "/.env/acpp-installdir"

    utils.acpp.clone_acpp(ACPP_GIT_DIR)

    ENV_SCRIPT_PATH = builddir+"/activate"

    ENV_SCRIPT_HEADER = ""
    ENV_SCRIPT_HEADER += "export SHAMROCK_DIR="+shamrockdir+"\n"
    ENV_SCRIPT_HEADER += "export BUILD_DIR="+builddir+"\n"
    ENV_SCRIPT_HEADER += "export ACPP_GIT_DIR="+ACPP_GIT_DIR+"\n"
    ENV_SCRIPT_HEADER += "export ACPP_BUILD_DIR="+ACPP_BUILD_DIR+"\n"
    ENV_SCRIPT_HEADER += "export ACPP_INSTALL_DIR="+ACPP_INSTALL_DIR+"\n"


    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export CMAKE_GENERATOR=\""+cmake_gen+"\"\n"
    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export MAKE_EXEC="+gen+"\n"
    ENV_SCRIPT_HEADER += "export MAKE_OPT=("+gen_opt+")\n"

    ENV_SCRIPT_HEADER += "export SHAMROCK_BUILD_TYPE=\""+cmake_build_type+"\"\n"
    ENV_SCRIPT_HEADER += "export SHAMROCK_CXX_FLAGS=\" --acpp-targets='"+acpp_target+"'\"\n"

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))
    source_file = "env_built_acpp.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../"+source_file))

    utils.envscript.write_env_file(
        source_path = source_path, 
        header = ENV_SCRIPT_HEADER, 
        path_write = ENV_SCRIPT_PATH)

    if is_acpp_already_installed(ACPP_INSTALL_DIR):
        print("-- acpp already installed => skipping")
    else:
        print("-- running compiler setup")
        os.system("bash -c 'cd "+builddir+" && . ./activate &&  updatecompiler'")

