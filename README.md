<picture>
   <source media="(prefers-color-scheme: dark)" srcset="doc/shamrock-doc/src/images/no_background_nocolor.png"  width="600">
   <img alt="text" src="doc/logosham_white.png" width="600">
 </picture>
 
![badge1](https://github.com/tdavidcl/Shamrock/actions/workflows/on_push_main.yml/badge.svg?branch=main) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# The Shamrock code

Shamrock is a general purpose HPC hydrodynamics simulation code focused on astrophysical contexts.  
The whole code is **C++17** by default, and all accelerated parts are done using **SYCL**, 
which can be directly compiled directly to native **CUDA**, **ROCM**, **OpenMP** and much more.
Shamrock also supports multiple GPUs and Heterogeneous clusters using **MPI**.  
This code aims to be: 
- Modern
- Modular  
- Fast 
- Portable
  
## Getting in touch

Join us on [Discord](https://discord.gg/Q69s5buyr5)! Alternatively, open a discussion or issue in this repository.

## Contributing

Shamrock accept contributions through github pull request :
1. Code contributions via [Pull request](https://github.com/tdavidcl/Shamrock/compare)
2. Documentation contributions via [Pull request](https://github.com/tdavidcl/Shamrock/compare)
3. Issue report & feature requests via [Github issues](https://github.com/tdavidcl/Shamrock/issues/new/choose)

If you want to contribute please fork the code and submit your pull requests from your fork.

## Citing the code

The day MNRAS will accept the paper 😅

## Compiler support

Compiler config | Support 
---|---
DPC++ CUDA | ![badge2](https://badgen.net/static/DPC++%2FCUDA/yes/green)  
DPC++ ROCM | ![badge2](https://badgen.net/static/DPC++%2FHIP:ROCM/yes/green)  
AdaptiveCPP OpenMP | ![badge2](https://badgen.net/static/ACPP%2FOpenMP/yes/green)  
AdaptiveCPP ROCM | ![badge2](https://badgen.net/static/ACPP%2FROCM/yes/green)  
AdaptiveCPP CUDA | ![badge2](https://badgen.net/static/ACPP%2FCUDA/yes/green)  
AdaptiveCPP SSCP | ![badge2](https://badgen.net/static/ACPP%2FSSCP/yes/green)  


# Documentation

We provide both a book like documentation and the more classic doxygen style for more details about the sources
 - The documentation is available here: [mkdocs doc](https://tdavidcl.github.io/Shamrock/mkdocs/index.html)
 - The doxygen doc is available here: [doxygen](https://tdavidcl.github.io/Shamrock/doxygen/index.html)

# Getting started

The whole getting started guide can be found here: [Getting started](https://tdavidcl.github.io/Shamrock/mdbook/usermanual/quickstart.html)

Note that a convenient way to pull the Shamrock repo is the following command to also pull the submodules:
```bash
git clone --recurse-submodules git@github.com:tdavidcl/Shamrock.git
```
