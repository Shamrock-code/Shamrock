import glob
import sys

file_list = glob.glob("log_precommit_*",recursive=True)

def load_file(fname):
    f = open(fname,'r')
    source = f.read()
    f.close()
    return source 

dic = []

print("# Pre commit report")
print()
print("Some failures were detected in pre-commit checks.\n Check the `On PR / Linting / Pre-commit CI (pull_request)` job in the tests for more detailled output")
print()
for f in file_list:
    #print(f)
    log_f = load_file(f)
    #print(log_f)

    if f == "log_precommit_check_sycl_include":
        print(log_f)
    elif f == "log_precommit_license_check":
        print(log_f)
    elif f == "log_precommit_pragma_once_check":
        print(log_f)
    elif f == "log_precommit_doxygen_header":
        print(log_f)
    else:
        print("```")
        print(log_f)
        print("```")


print("## Suggested changes")
print("<details>")
print("<summary>")
print("Detailed changes :")
print("</summary>")
print(f" ")
print("```diff")
print(load_file("diff-pre-commit"))
print("```")
print("")
print("</details>")