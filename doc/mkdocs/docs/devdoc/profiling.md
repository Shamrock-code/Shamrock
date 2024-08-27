# Profiling

## Shamrock custom profiling

turned on by cmake : `SHAMROCK_USE_PROFILING=On`

In the code many function starts with

```c++
StackEntry stack_loc{};
```
the profiling can be disabled for a function by using :
```c++
StackEntry stack_loc{false};
```

This is used initially to trace the location in the code, allowing more precise error message, but also profiling !

when the code is ran some files name `timings_*` will be created, they hold the flame graph tracing of the code.

```bash
python buildbot/merge_profilings.py timings_* merge_prof
```

This will merged the output of each files.

Go on `https://ui.perfetto.dev/` and upload the `merge_prof` file you will see the actual trace of the code execution.

## NVTX profiling

Shamrock can also use NVTX based tooling, enabled by `SHAMROCK_USE_NVTX=On` in Cmake

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
nsys profile -t cuda,nvtx,mpi --gpu-metrics-device=1,2,3,4 --cuda-memory-usage=true --mpi-impl=openmpi  mpirun -n 4 ./shamrock --sycl-cfg auto:CUDA --sycl-ls-map --loglevel 1 --rscript ../exemples/spherical_wave.py
```

### NCU

```
ncu --set full --call-stack --nvtx --section=SpeedOfLight_HierarchicalDoubleRooflineChart --section=SpeedOfLight_HierarchicalSingleRooflineChart --section=SpeedOfLight_HierarchicalTensorRooflineChart --open-in-ui ./shamrock --sycl-cfg 1:1 --loglevel 10 --rscript ../../exemples/spherical_wave.py
```
