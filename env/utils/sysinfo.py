
import os
import shutil

def is_ninja_available():
    return not (shutil.which("ninja") == None)


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