import argparse
import os
import utils.intel_llvm
import utils.sysinfo
import utils.envscript
import utils.cuda_arch
import utils.amd_arch

NAME = "Debian generic Intel LLVM"
PATH = "machine/debian-generic/intel-llvm"

print("loading :",NAME)

def is_intel_llvm_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/clang++")

def setup(argv,builddir, shamrockdir,buildtype):

    print("------------------------------------------")
    print("Running env setup for : "+NAME)
    print("------------------------------------------")

    parser = argparse.ArgumentParser(prog=PATH,description= NAME+' env for Shamrock')

    parser.add_argument("--target", action='store', help="sycl backend to use")
    parser.add_argument("--gen", action='store', help="generator to use (ninja or make)")
    
    parser.add_argument("--cuda", action='store_true', help="build intel llvm with cuda support")
    parser.add_argument("--cuda-path", action='store', help="set cuda path")

    parser.add_argument("--hip", action='store_true', help="build intel llvm with hip support")
    parser.add_argument("--rocm-path", action='store', help="set ROCM path")

    args = parser.parse_args(argv)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)
    
    INTELLLVM_GIT_DIR = builddir+"/.env/intel-llvm-git"
    INTELLLVM_INSTALL_DIR = builddir + "/.env/intel-llvm-installdir"

    utils.intel_llvm.clone_intel_llvm(INTELLLVM_GIT_DIR)

    configure_args = utils.intel_llvm.get_llvm_configure_arg(args)  
    shamcxx_args = utils.intel_llvm.get_intel_llvm_target_flags(args)

    ENV_SCRIPT_PATH = builddir+"/activate"

    ENV_SCRIPT_HEADER = ""
    ENV_SCRIPT_HEADER += "export SHAMROCK_DIR="+shamrockdir+"\n"
    ENV_SCRIPT_HEADER += "export BUILD_DIR="+builddir+"\n"
    ENV_SCRIPT_HEADER += "export INTELLLVM_GIT_DIR="+INTELLLVM_GIT_DIR+"\n"
    ENV_SCRIPT_HEADER += "export INTELLLVM_INSTALL_DIR="+INTELLLVM_INSTALL_DIR+"\n"
    ENV_SCRIPT_HEADER += "export INTELLLVM_CONFIGURE_ARGS=("+configure_args+")\n"
    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export CMAKE_GENERATOR=\""+cmake_gen+"\"\n"
    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export MAKE_EXEC="+gen+"\n"
    ENV_SCRIPT_HEADER += "export MAKE_OPT=("+gen_opt+")\n"
    ENV_SCRIPT_HEADER += "export SHAMROCK_CXX_FLAGS=\""+shamcxx_args+"\"\n"
    ENV_SCRIPT_HEADER += "export SHAMROCK_BUILD_TYPE=\""+cmake_build_type+"\"\n"
    ENV_SCRIPT_HEADER += "\n"

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))
    source_file = "env_built_intel-llvm.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../"+source_file))

    utils.envscript.write_env_file(
        source_path = source_path, 
        header = ENV_SCRIPT_HEADER, 
        path_write = ENV_SCRIPT_PATH)

    if is_intel_llvm_already_installed(INTELLLVM_INSTALL_DIR):
        print("-- intel llvm already installed => skipping")
    else:
        print("-- running compiler setup")
        os.system("bash -c 'cd "+builddir+" && source ./activate &&  updatecompiler'")