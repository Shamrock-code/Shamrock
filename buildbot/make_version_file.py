import os
import sys

strfile = ""

try:
    fvers = open(sys.argv[1],'r')
    strfile = fvers.read()
    fvers.close()
except :
    None



str_git = os.popen("git log -n 1 --decorate=full").read()
git_hash = str_git.split()[1]
git_head = str_git[str_git.find("HEAD -> ")+8:str_git.find(")")]



str_ = '''
#include "shamrock/version.hpp"

const std::string git_info_str = R"%%('''

str_ += "     commit : "+git_hash+"\n"
str_ += "     HEAD   : "+git_head+"\n"
str_ += "     modified files (since last commit):"+"\n"
str_ += os.popen("git diff-index --name-only HEAD -- | sed \"s/^/        /g\"").read()
str_ += r')%%";'
str_ += "\n\n"

str_ += "const std::string git_commit_hash = \""+git_hash+"\";\n"

str_ += "const std::string compile_arg = \""+sys.argv[2]+"\";\n"

str_ += "const std::string version_string = \""+sys.argv[3]+"\";\n"


if not (strfile == str_):
    f = open(sys.argv[1],'w')
    f.write(str_)
    f.close()

# else:
#     print("no change in version.cpp")
