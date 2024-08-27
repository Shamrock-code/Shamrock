# Environment config in Shamrock

In Shamrock to ease configuration we provide environment script, the goal is for them to ease setup on known machines.

- Activate file with exports in build folder, configure environment behavior through exports and register functions
- Helper script to generate the env : env/new-env

Functions registered by the env :
- shammake that call ninja or make automatically
- shamconfigure that call cmake and forward argument to the cmake call

Typicall workflow would be :
```
env/new-env --machine psmn-cascade --builddir build
cd build
. activate
shamconfigure
shammake
```


## new env script

```sh
> env/new-env --help
usage: new-env [-h] --machine MACHINE --builddir BUILDDIR -- (argument for the env)

Environment utility for Shamrock

options:
  -h, --help           show this help message and exit
  --machine MACHINE    machine assumed for the environment
  --builddir BUILDDIR  build directory to use
  --                   Everything after this will be forwarded to the env.
```

This call will print the help of the debian-generic environment
`env/new-env --machine debian-generic --builddir build -- --help `
