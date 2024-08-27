# Code coverage


## Configuration

For now the code coverage has only been done with opensycl with profile `omp_coverage`.

```sh
python buildbot/configure.py --gen ninja --build debug --tests --outdir build_opensycl_cov --cxxpath ../sycl_cpl/OpenSYCL --compiler opensycl --profile omp_coverage
```

## Running the test and generating reports

to generate the code coverage done by the tests :
```sh
LLVM_PROFILE_FILE="utests.profraw" ./shamrock_test --sycl-cfg 0:0 --loglevel 0 --unittest
llvm-profdata merge -sparse utests.profraw -o utests.profdata
```

to print a report to the terminal:
```sh
llvm-cov report shamrock_test -instr-profile=utests.profdata
```

to dump the report to an html file
```sh
llvm-cov show shamrock_test -instr-profile=utests.profdata -format=html -output-dir=out_cov -Xdemangler c++filt -Xdemangler -n -ignore-filename-regex=".*\Tests.cpp$|.*\Tests.hpp$|.*\shamtest.cpp|.*\shamtest.hpp|.*\main_test.cpp|.*\aliases.hpp"
```
