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

### Step 0 - Check the already existing build folder

Check if a folder does not already exist before create and env. If you found an existing folder (e.g. activate script present in it) go to step 4

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
./shamenv_do shamconfigure     # alias to the correct cmake command
./shamenv_do shammake && echo "build done"        # alias to ninja build (or make if ninja is unavailable), the echo part allows the llm to understand that the build succedeed even if it does not show 100% completion as ninja does sometimes
```

## Testing

```bash
cd build
./shamenv_do ./shamrock --smi          # or ./shamrock_test --smi
```

Pick a device ID from the output, then run:

```bash
./shamenv_do ./shamrock_test --sycl-cfg <id>:<id> --loglevel 1 --unittest
```

where `<id>` is the device ID the user selected from `--smi` output.

## Code style & linting

- **Formatter**: `.clang-format`
- **CI linter**: `.clang-tidy`
- **Pre-commit hooks**: `.pre-commit-config.yaml`
- Run `pre-commit run --all-files` before committing

## Naming conventions (from `.clang-tidy` `CheckOptions`)

| Entity                         | Case       |
| ------------------------------ | ---------- |
| Class/Enum/Union               | CamelCase  |
| Function/Variable/Parameter    | lower_case |
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

## Agent commit attribution

Agent-made commits should use `Assisted-by: <agent_name>` instead of
`Co-Authored-by`. Reserve `Co-Authored-by` for human collaborators only.

## Upstream repo & PRs

The upstream repo is `Shamrock-code/Shamrock`.
PR lookups should target the upstream:

```bash
gh pr list --repo Shamrock-code/Shamrock
gh pr view <number> --repo Shamrock-code/Shamrock
```

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
pwd && ls && cd build && ./shamenv_do shammake && echo "build done"

# Run pre-commit
pre-commit run --all-files

# Run tests
pwd && ls && cd build && ./shamenv_do ./shamrock_test --sycl-cfg <id>:<id> --loglevel 1 --unittest
```
