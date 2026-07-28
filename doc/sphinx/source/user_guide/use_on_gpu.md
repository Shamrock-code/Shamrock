# Using Shamrock on GPUs

Ok, so it's quite hard to list all possibilities, so for the time being I'll assume that you already have CUDA, ROCM, OpenCL, or Level-zero installed.

In general, if you are using a machine with a Shamrock env where there is already support for it, job done, just do as we do in the quickstart except that you will select your GPU.

If not, well I would recommend going with the AdaptiveCpp SSCP backend, it will automatically recompile the kernels for the selected hardware at runtime, so it is probably the simplest option. If you want to see alternatives, look at specific machines such as Argonne Aurora and CBP DGX, copy-paste the environments, and modify them for your specific system.

- For CUDA, go there [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux) and go to the [Package manager section](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation).
- For AMD, to install ROCM go there [link](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) and go to the [Package manager section](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html).
- For Intel, go to the oneAPI install page [link](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-2/overview.html).

In any case, once it is installed, if you are on Debian or Ubuntu use the following env setup:

```bash
./env/new-env --machine debian-generic.acpp --builddir build -- --backend generic
shamconfigure
shammake
```

Normally the GPU backend should be automatically detected. If not, edit the `activate` file in the build folder and look for that section toward the env:

```bash
function setupcompiler {
    clone_acpp || return
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} ${CCACHE_CMAKE_ARG} -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} || return
    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}
```

- If CUDA is not detected add `-DWITH_CUDA_BACKEND=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda` to the command.
- If ROCM is not detected add `-DWITH_ROCM_BACKEND=ON -DROCM_PATH=/path/to/rocm` to the command.

For OpenCL and Level-zero it should be automatic. If you want to debug all of that, the AdaptiveCpp documentation details the process on several hardware setups [link](https://adaptivecpp.github.io/AdaptiveCpp/).

You can check if you have support for a GPU by doing `./shamrock --smi` and then run `./shamrock --smi --benchmark-mpi --sycl-cfg 0:0` while selecting the GPU. If it works you are good to go.

If you are a bit lost or can not get it to work ask on [Discord](https://discord.gg/Q69s5buyr5) and we'll try to fix and also fill holes in the documentation for your current setup.
