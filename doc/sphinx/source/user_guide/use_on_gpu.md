# Using Shamrock on GPUs

Ok, so it's quite hard to list all possibilities, so for the time being I'll assume that you already have CUDA, ROCM, OpenCL, or Level-zero installed.

## Installing the GPU toolkit

If the GPU toolkit for your hardware is not installed yet, here is where to start:

- For CUDA, go there [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux) and go to the [Package manager section](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation).
- For AMD, to install ROCM go there [link](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) and go to the [Package manager section](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html).
- For Intel, go to the oneAPI install page [link](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-2/overview.html).

## Choosing a build environment

### Machines with an existing Shamrock environment

In general, if you are using a machine with a Shamrock env where there is already support for it, job done, just do as we do in the quickstart except that you will select your GPU.

### Other machines

If not, well I would recommend going with the AdaptiveCpp SSCP backend, it will automatically recompile the kernels for the selected hardware at runtime, so it is probably the simplest option (see the section below).

If you want to see alternatives, look at specific machines such as Argonne Aurora and CBP DGX, copy-paste the environments, and modify them for your specific system.

## Building with AdaptiveCpp SSCP

Once the GPU toolkit is installed, if you are on Debian or Ubuntu use the following env setup:

```bash
./env/new-env --machine debian-generic.acpp --builddir build -- --backend generic
shamconfigure
shammake
```

(gpu-backend-not-detected)=

### If the GPU backend is not detected

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

## Running on a GPU

Once Shamrock is built with GPU support, running on a GPU comes down to picking the right device with `--sycl-cfg`.

### Step 1: list the available devices

```bash
./shamrock --smi
```

This prints every SYCL device Shamrock can see. For example, on a machine with one NVIDIA GPU you would get something like:

```
| id |      Device name          |      Platform name     |  Type  |    Memsize   | units |
-------------------------------------------------------------------------------------------
|  0 |   AdaptiveCpp OpenMP host |     AdaptiveCpp OpenMP |    CPU |     62.00 GB |    16 |
|  1 |   NVIDIA GeForce RTX 3070 |      CUDA (platform 0) |    GPU |      7.78 GB |    46 |
```

Note the `id` of your GPU (`1` here). If your GPU is not listed, the corresponding backend was not built, see {ref}`gpu-backend-not-detected`.

### Step 2: select the device

Pass the device id twice to `--sycl-cfg`:

```bash
./shamrock --sycl-cfg 1:1 --loglevel 1 --rscript my_script.py
```

The two numbers select the device used for compute and an auxiliary queue. Unless you know you want otherwise, use the same id for both.

If you use Shamrock as a Python package, the equivalent is:

```python
import shamrock

if not shamrock.sys.is_initialized():
    shamrock.sys.init("1:1")
```

### Step 3: check that it works

```bash
./shamrock --smi --benchmark-mpi --sycl-cfg 1:1
```

The `Selected devices` block should show your GPU, and the benchmark should run to completion. If it does, you are good to go.

### Vendor-specific notes

#### NVIDIA (CUDA)

- The platform name contains `CUDA`.
- Restrict which GPUs are visible with `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=0,1`.

#### AMD (ROCm / HIP)

- The platform name contains `HIP`.
- Restrict which GPUs are visible with `ROCR_VISIBLE_DEVICES` (or `HIP_VISIBLE_DEVICES`).
- On MI250X, each GCD shows up as a separate device.

#### Intel (Level Zero / OpenCL)

- The GPU may show up twice, once through Level Zero and once through OpenCL. Prefer the Level Zero entry.
- Restrict which GPUs are visible with `ZE_AFFINITY_MASK`. On Aurora-class GPUs, `ZE_FLAT_DEVICE_HIERARCHY=FLAT` exposes each tile as its own device (see the scripts in `env/machine/argonne/aurora/`).

### Using several GPUs with MPI

Run one MPI rank per GPU and let Shamrock pick the device of each rank with `auto:<platform>`:

```bash
mpirun -n 4 ./shamrock --sycl-cfg auto:CUDA --loglevel 1 --smi --rscript my_script.py
```

With `auto:<name>`, Shamrock keeps only the devices whose platform name contains `<name>` (`CUDA`, `HIP`, ...), and each rank takes the device at index `local_rank % n_matching_devices`. The node-local rank is read from the environment set by the MPI launcher (Open MPI, MVAPICH2, Intel MPI, Slurm, PALS). If none of these is detected, `auto:` fails with `cannot query local rank`.

Use `--smi-full` instead of `--smi` to print which device each rank ended up on. The `exemple_batch.sh` scripts in `env/machine/` (e.g. Adastra, LUMI) show working multi-GPU setups.

### Troubleshooting

- **My GPU is not listed by `--smi`**: the backend was not compiled in, check the AdaptiveCpp configuration (see {ref}`gpu-backend-not-detected`).
- **`the compute queue id is larger than the number of queue`**: the id passed to `--sycl-cfg` does not exist. Re-run `--smi` and check the id.
- **It runs, but on the CPU**: you picked the OpenMP/host device id. Check the `Type` column of `--smi`.
- **Still stuck?** See {ref}`gpu-getting-help`.

(gpu-getting-help)=

## Getting help

If you are a bit lost or can not get it to work ask on [Discord](https://discord.gg/Q69s5buyr5) and we'll try to fix and also fill holes in the documentation for your current setup.
