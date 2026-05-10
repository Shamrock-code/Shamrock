# SHAMROCK — Project Guide

## What is this project

SHAMROCK is a C++17 hydrodynamics framework built with SYCL, MPI,
and Python. SYCL backend support covers every major implementation
(AdaptiveCpp, DPC++, intel/llvm). It uses CMake (generators: make or
Ninja depending on availability) and has submodules.

## Building

A "machine" is one OS + hardware combination.
Run `./env/new-env` without arguments to see the
full list of available machine configurations.

### Step 1 — Select a machine

```bash
./env/new-env
```

Pick one from the list (e.g. `debian-generic.acpp` on Debian).

### Step 2 — Inspect machine-specific options

```bash
./env/new-env --machine <selected machine> --builddir build -- --help
```

This shows the flags specific to that machine — they can vary widely.

### Step 3 — Create the environment

```bash
./env/new-env --machine <selected machine> --builddir build -- \
  <machine specific flags>
```

### Step 4 — Build

```bash
cd build
source ./activate
shamconfigure     # alias to the correct cmake command
shammake          # alias to ninja build (or make if ninja is unavailable)
```

## Testing

```bash
cd build
source ./activate
./shamrock --smi          # or ./shamrock_test --smi
```

Pick a device ID from the output, then run:

```bash
./shamrock_test --sycl-cfg <id>:<id> --loglevel 1 --unittest
```

where `<id>` is the device ID the user selected from `--smi` output.

## Code style & linting

- **Formatter**: `.clang-format`
- **CI linter**: `.clang-tidy`
- **Pre-commit hooks**: `.pre-commit-config.yaml`
- Run `pre-commit run --all-files` before committing

## Naming conventions (from `.clang-tidy` `CheckOptions`)

| Entity                         | Case       |
| ------ ------ ---------------- | ---------- |
| Class/Enum/Union               | CamelCase  |
| Function/Variable/Parameter/   | lower_case |
| Member                         | lower_case |

## Architecture overview

```text
src/
  shamalgs/          GPU & MPI algorithms
  shambackends/      SYCL GPU device management and kernels
  shambase/          base containers, math utils, I/O
  shambindings/      embeds Python via pybind11, registering C++ types and modules
  shamcmdopt/        CLI argument parsing, env/tty detection utilities
  shamcomm/          MPI and SYCL comm layer for Shamrock
  shammath/          tensor and linear algebra math routines
  shammodels/        SPH, GSPH, Ramses, Zeus hydro model implementations
  shamphys/          physics utilities: EOS, MHD, orbits, collapse
  shamrock/          core hydrodynamics framework: solvers, mesh, AMR, I/O, scheduler, graph
  shamsys/           SHAMROCK system and runtime glue
  shamtest/          Shamrock's internal C++ test framework
  shamtree/          SYCL-accelerated Morton-code trees for hydrodynamics queries
  shamunits/         compile-time physics unit conversion library
  pylib/             Python package root for Shamrock
  tests/             unit tests for Shamrock library components
```

## Files to avoid modifying unless explicitly asked

- `.github/workflows/*.yml` — CI workflows.
- `external/` submodules — upstream dependencies.
- `LICENSE`, `LICENSE.en` — legal files.

## Quick reference: common commands

```bash
# List available machines
./env/new-env

# Inspect machine-specific options
./env/new-env --machine <machine> --builddir build -- --help

# Configure for development
./env/new-env --machine <machine> --builddir build-debug -- \
  <machine specific flags>

# Build
pwd && ls && cd build && source ./activate && shammake

# Run pre-commit
pre-commit run --all-files

# Run tests
pwd && ls && cd build && source ./activate && ./shamrock_test --sycl-cfg <id>:<id> --loglevel 1 --unittest
```
