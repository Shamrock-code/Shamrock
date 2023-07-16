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
nsys profile --gpu-metrics-device=0 ./shamrock --sycl-cfg 1:1 --loglevel 1 --rscript ../../exemples/spherical_wave.py
```