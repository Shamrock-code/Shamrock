# Profiling

## Shamrock custom profiling

### Basics

In Shamrock multiple tools are available to profile the code.
In particular most of the tools are enabled by setting the cmake option `SHAMROCK_USE_PROFILING=On` (which is on by default).

This enables the use of the following environment variables  :

- `SHAM_PROFILING` : Enable Shamrock profiling
- `SHAM_PROF_PREFIX` : Prefix of shamrock profile outputs
- `SHAM_PROF_USE_NVTX` : Enable NVTX profiling
- `SHAM_PROF_USE_COMPLETE_EVENT` :Use complete event instead of begin end for chrome tracing
- `SHAM_PROF_EVENT_RECORD_THRES` : Change the event recording threshold

For Shamrock compiled with profiling enabled you have many options availables.
First of by default nothing appends and the profiling overhead should be low enough to be ignored. If you want Shamrock to generate profiling flag you should set the env variable `SHAM_PROFILING=1`. This enables the profiling dump to a file set by the env variable `SHAM_PROF_PREFIX`, which will be named as `${SHAM_PROF_PREFIX}.${MPI_WORLD_RANK}.json`.

After Shamrock has finished its job you can use the script `merge_profilings.py` to merge all the traces into a single one by doing
```bash
python buildbot/merge_profilings.py ${SHAM_PROF_PREFIX}.*
```
This will create a file `merged_profile.json` that can be viewed using either `chrome://tracing/` or [Perfetto UI](https://ui.perfetto.dev/).

### Options

The behavior of the profiling can be controlled using a few options. First `SHAM_PROF_EVENT_RECORD_THRES` env variable can be used to set the threshold time for event to be registered ($10 \mu s$ by default), any event shorter than this threshold won't be recorded.
Additionally setting it to `0` will record any event regardless of their duration.

The option `SHAM_PROF_USE_COMPLETE_EVENT` controls wether completed event or `begin` `end` events will be used in the chrome tracing dump.

Lastly the option `SHAM_PROF_USE_NVTX` will enable NVTX profiling in shamrock.

!!! warning "NVTX profiling"

    Be aware that both `SHAM_PROFILING` and `SHAM_PROF_USE_NVTX` must be set to `1` and that Shamrock must be compiled with the cmake option `SHAMROCK_USE_NVTX=On` for NVTX to work.

### Adding profiling entries in the code

In the code many function starts with

```c++
StackEntry stack_loc{};
```
the profiling can be disabled for a function by using :
```c++
StackEntry stack_loc{false};
```

This is used initially to trace the location in the code, allowing more precise error message, but also profiling !

## Nvidia profiling

### Nsys

for a timeline view, with GPU metrics:

```
nsys profile -t cuda,nvtx --gpu-metrics-device=0 ./shamrock --sycl-cfg 1:1 --loglevel 1 --rscript ../../exemples/spherical_wave.py
```

MPI version :
```
nsys profile -t cuda,nvtx,mpi --cuda-memory-usage=true --mpi-impl=openmpi ./shamrock --sycl-cfg 1:1 --loglevel 1 --rscript ../../exemples/spherical_wave.py
```

On the CBP (ENSL) the qstrm importer fails, bu it can be ran a posteriori :
```
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i <input> -o output.qdrep
```

 MPI trace :

```
nsys profile -t cuda,nvtx,mpi --cuda-memory-usage=true --mpi-impl=openmpi /usr/bin/mpirun -n 2 ./shamrock --sycl-cfg auto:CUDA --loglevel 1 --rscript ../../exemples/spherical_wave.py
```

Current command on the GDX :
```
nsys profile -t cuda,nvtx,mpi --gpu-metrics-device=1,2,3,4 --cuda-memory-usage=true --mpi-impl=openmpi  mpirun -n 4 ./shamrock --sycl-cfg auto:CUDA --smi-full --loglevel 1 --rscript ../exemples/spherical_wave.py
```

### NCU

```
ncu --set full --call-stack --nvtx --section=SpeedOfLight_HierarchicalDoubleRooflineChart --section=SpeedOfLight_HierarchicalSingleRooflineChart --section=SpeedOfLight_HierarchicalTensorRooflineChart --open-in-ui ./shamrock --sycl-cfg 1:1 --loglevel 10 --rscript ../../exemples/spherical_wave.py
```
