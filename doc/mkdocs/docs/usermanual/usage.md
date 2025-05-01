# Shamrock usage


## Running the main executable

If you run `./shamrock --help`, you will see the following :

```
executable : ./shamrock

Usage :
--benchmark-mpi                 : micro benchmark for MPI
--color                         : force colored ouput
--feenableexcept                 : Enable FPE exceptions
--force-dgpu-off                 : for direct mpi comm off
--force-dgpu-on                 : for direct mpi comm on
--help                          : show this message
--ipython                       : run shamrock in Ipython mode
--loglevel      (logvalue)      : specify a log level
--nocolor                       : disable colored ouput
--pypath        (sys.path)      : python sys.path to set
--pypath-from-bin (python binary) : set sys.path from python binary
--rscript       (filepath)      : run shamrock with python runscirpt
--smi                           : print information about available SYCL devices in the cluster
--smi-full                      : print information about EVERY available SYCL devices in the cluster
--sycl-cfg      (idcomp:idalt)  : specify the compute & alt queue index

Env variables :
  SHAM_PROF_PREFIX              : Prefix of shamrock profile outputs
  SHAM_PROF_USE_NVTX            : Enable NVTX profiling
  SHAM_PROFILING                : Enable Shamrock profiling
  SHAM_PROF_USE_COMPLETE_EVENT  : Use complete event instead of begin end for chrome tracing
  SHAM_PROF_EVENT_RECORD_THRES  : Change the event recording threshold
  NO_COLOR                      : Disable colors (if no color cli args are passed)
  CLICOLOR_FORCE                : Enable colors (if no color cli args are passed)
  TERM                          : Terminal emulator identifier
    = xterm-256color
  COLORTERM                     : Terminal color support identifier
    = truecolor
  SHAMTTYCOL                    : Set tty assumed column count

Env deduced vars :
  isatty = Yes
  color = enabled
  tty size = 36x106
```

Most of those options are just changing the configuration of the code at runtime and will be explained later.

To start you must find a configuration that works try running `./shamrock --sycl-cfg 0:0`, which target sycl device 0 with all the queues (see later for detailed explanation of the queue configurations).

If everything works you should see something like :

```

  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░

-----------------------------------------------------

Git infos :
     commit : 8e5baddbc044867ebd42754410d186eb319d9ede
     HEAD   : refs/heads/feature/htol-bump-init, refs/remotes/origin/feature/htol-bump-init
     modified files (since last commit):
        external/pybind11

-----------------------------------------------------
MPI status :
 - MPI & SYCL init : Ok
 - MPI CUDA-AWARE : Yes
 - MPI ROCM-AWARE : Unknown
 - MPI use Direct Comm : Yes
 - MPI use Direct Comm : Working
-----------------------------------------------------
log status :
 - Loglevel : 0 , enabled log types :
     [xxx] : xxx ( logger::normal )
     [xxx] Warning : xxx ( logger::warn )
     [xxx] Error : xxx ( logger::err )
-----------------------------------------------------
 - Code init DONE now it's time to ROCK
-----------------------------------------------------
-----------------------------------------------------
 - MPI finalize
Exiting ...

 Hopefully it was quick :')
```

Which mean that the executable did run, and the self check worked, but you didn't ask anything to shamrock.

## Running in Ipython mode

Assuming you are running on device 0 still, to run in ipython mode do simply : `./shamrock --sycl-cfg 0:0`, which should give a similar output as previously but ending with :
```
--------------------------------------------
-------------- ipython ---------------------
--------------------------------------------
SHAMROCK Ipython terminal
Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0]

###
import shamrock
###

In [1]:
```

From that point just type `ìmport shamrock` in that Ipython terminal to initialise the python interoperability.

For exemple you can querry informations about the status of the code :
```py
In [1]: import shamrock

In [2]: print(shamrock.get_git_info())
     commit : 8e5baddbc044867ebd42754410d186eb319d9ede
     HEAD   : refs/heads/feature/htol-bump-init, refs/remotes/origin/feature/htol-bump-init
     modified files (since last commit):
        external/pybind11
```

## Running a runscript

An other possibility (the one that is the most used also) is to start a runscript, which is just a python script starting that will be executed by shamrock, for exemple this is a very basic script that get the SPH model with M6 kernel and just start the patch scheduler on it.

```py
import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_eos_locally_isothermal()
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)
```

## Running on multiple nodes

For exemple on a cluster using OARsub the command to run

```sh
mpirun -machinefile $OAR_NODEFILE --bind-to socket -npernode 2 sh runscript.sh
```

Here `mpirun` is the standard command to start a program using MPI (to start multiple process that can work together using MPI library). `-machinefile` is used to specify the layout of the nodes that will be used by MPI (if you have reserved 10 nodes for exemple this is the list of nodes available). Starting from OpenMPI 4 `--bind-to` specify the list of CPU cores that are used by each process started by MPI, here `--bind-to socket` tell MPI that each process should be attach to a socket. On the machine used is the exemple there are 2 CPUs per nodes therefor to start one process per CPU one should use `--bind-to socket -npernode 2`, where a socket refer to the "slot" where the CPU is installed (2 on this system). Here `sh runscript.sh` is just the command launched by each process here runscript is :
```sh
export ACPP_DEBUG_LEVEL=0
export LD_LIBRARY_PATH=/grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/llvm-13.0.1-i53qugtbmlvnfi6tppnc7bresushxg2j/lib:$LD_LIBRARY_PATH
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=32

./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ../exemples/spherical_wave.py
```

Sadly hardware and software may differ greatly from clusters to clusters see [Cluster](cluster.md) for details on known clusters.
