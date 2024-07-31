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

        print("## ❌ Check SYCL `#include`")
        print("""

The pre-commit checks have found some #include of non-standard SYCL headers

It is recommended to replace instances of `#include <hipSYCL...` by `#include <shambackends/sycl.hpp>` 
which will include the sycl headers.

At some point we will refer to a guide in the doc about this

        """)
    if f == "log_precommit_license_check":

        print("## ❌ Check license headers")
        print("""

The pre-commit checks have found some missing or ill formed license header.
All C++ files (headers or sources) should start with : 
```
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
```
Any line break before this header or change in its formatting will trigger the fail of this test
        """)

    if f == "log_precommit_pragma_once_check":

        print("## ❌ Check #pragma once")
        print("""

The pre-commit checks have found some headers that are not starting with `#pragma once`.
This indicates to the compiler that this header should only be included once per source files avoid double definitions of function or variables

All headers files should have, just below the license header the following line :
```
#pragma once
```

At some point we will refer to a guide in the doc about this

        """)
        
    if f == "log_precommit_doxygen_header":
        print(log_f)


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