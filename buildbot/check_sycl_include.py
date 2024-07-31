ignore_list = [
    "src/shambackends/include/shambackends/sycl.hpp",
    #"src/shambackends/include/shambackends/typeAliasFp16.hpp"
]


from lib.buildbot import * 
import glob
import sys
import re

print_buildbot_info("SYCL #include check tool")

file_list = glob.glob(str(abs_src_dir)+"/**",recursive=True)

file_list.sort()

def should_check_file(fname):
    is_hpp = fname.endswith(".hpp")
    is_cpp = fname.endswith(".cpp")
    is_ign = fname.replace(abs_proj_dir+ "/", "") in ignore_list
    return (is_hpp or is_cpp) and (not is_ign)

def load_file(fname):
    f = open(fname,'r')
    source = f.read()
    f.close()
    return source

def write_file(fname, source):
    f = open(fname,'w')
    f.write(source)
    f.close()

def should_corect(source):
    if "#include <hipSYCL" in source:
        return True

def autocorect(source):
    source = re.sub(r"#include <hipSYCL(.+)\n", r"#include <shambackends/sycl.hpp>\n",source)
    return source
        

has_found_errors = False

for fname in file_list:
    if (should_check_file(fname)):
        source = load_file(fname)

        if should_corect(source):
            
            if not has_found_errors:
                print(" => \033[1;34mNon standard SYCL #include found \033[0;0m: ")
                print("The check found so instances of sycl inclusion using non standard headers")
                print("Please remove instances of :")
                print("  #include <hipSYCL/*")
                print()
                print("Trying autocorect : ")
                has_found_errors = True

            print(" -",fname)
            source = autocorect(source)

            write_file(fname, source)


if has_found_errors:
    print("Autocorect done !")
    print()
    sys.exit("Exitting with check failure")
else : 
    print(" => \033[1;34mSYCL #includes status \033[0;0m: OK !")
