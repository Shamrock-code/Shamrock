import argparse
import shutil
import os

NAME = "Debian generic AdaptiveCpp"
PATH = "machine/debian-generic/acpp"

print("loading :",NAME)


def clone_acpp(folder):
    if os.path.isdir(folder):
        print("-- skipping git clone folder does already exist")
    else:
        print("-- clonning https://github.com/AdaptiveCpp/AdaptiveCpp.git")
        os.system("git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git "+folder)

def is_acpp_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/acpp")

def is_ninja_available():
    return not (shutil.which("ninja") == None)

def write_env_file(source_file, header, path_write):

    ENV_SCRIPT_CONTENT = header + "\n"

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    with open(os.path.abspath(os.path.join(cur_file, "../"+source_file))) as f:
        contents = f.read()
        ENV_SCRIPT_CONTENT += contents 

    with open(path_write, "w") as env_script:
        env_script.write(ENV_SCRIPT_CONTENT)


def get_avail_mem():
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    return free_m

def should_limit_comp_cores():
    MAX_COMP_SZ = 1e9
    avail = get_avail_mem()*1e6

    limit = False
    cnt = os.cpu_count()

    avail_per_cores = avail / os.cpu_count()
    if avail_per_cores < MAX_COMP_SZ:
        print("-- low memory per cores, limitting number of thread for compilation")
        print("   ->  free memory /cores :", avail / os.cpu_count())
        cnt = int(avail / MAX_COMP_SZ)
        limit = True
        if cnt < 1:
            cnt = 1
        print("   ->  limiting to", cnt,"cores")

    return limit,cnt




def select_generator(args):

    limit_cores, cores = should_limit_comp_cores()

    gen = "make"
    gen_opt = ""

    if args.gen == None:
        if is_ninja_available():
            gen = "ninja"
    else:
        gen = args.gen

    cmake_gen = ""
    if gen == "make":
        cmake_gen = "Unix Makefiles"
        gen_opt = " -j "+str(cores)
    elif gen == "ninja":
        cmake_gen = "Ninja"
        if limit_cores:
            gen_opt = " -j "+str(cores)
        else:
            gen_opt = ""
    else:
        raise "unknown generator "+gen

    if args.gen == None:
        print("-- generator not specified, defaulting to :",gen)

    return gen, gen_opt, cmake_gen


def setup(argv,builddir, shamrockdir):

    print("------------------------------------------")
    print("Running env setup for : "+NAME)
    print("------------------------------------------")

    parser = argparse.ArgumentParser(prog=PATH,description= NAME+' env for Shamrock')

    parser.add_argument("--backend", action='store', help="sycl backend to use")
    parser.add_argument("--arch", action='store', help="arch to build")
    parser.add_argument("--gen", action='store', help="generator to use (ninja or make)")

    args = parser.parse_args(argv)

    gen, gen_opt, cmake_gen = select_generator(args)
    
    ACPP_GIT_DIR = builddir+"/.env/acpp-git"
    ACPP_BUILD_DIR = builddir + "/.env/acpp-builddir"
    ACPP_INSTALL_DIR = builddir + "/.env/acpp-installdir"

    clone_acpp(ACPP_GIT_DIR)

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
    ENV_SCRIPT_HEADER += "export MAKE_OPT=\""+gen_opt+"\"\n"

    write_env_file(
        source_file = "env_built_acpp.sh", 
        header = ENV_SCRIPT_HEADER, 
        path_write = ENV_SCRIPT_PATH)

    if is_acpp_already_installed(ACPP_INSTALL_DIR):
        print("-- acpp already installed => skipping")
    else:
        print("-- running compiler setup")
        os.system("sh -c 'cd "+builddir+" && source ./activate &&  updatecompiler'")

