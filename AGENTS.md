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

Check if a folder with a `shamenv_do` script already exists. If it does, **do not re-run `./env/new-env`** — the environment is already configured. Skip directly to Step 4 (Build). Only run `./env/new-env` when creating a brand-new build directory.

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
./shamenv_do shamconfigure # alias to the correct cmake command
./shamenv_do shammake      # alias to ninja build (or make if ninja is unavailable)
```

Always use something like `&& echo DONE` after the build command to avoid confusion since `ninja` sometimes can do a successful build without showing 100% in the steps.

Check if `./shamrock`, `./shamrock_test` are present in the build dir, if yes it has succeeded.

## Testing

**BEFORE running any unittest, always check that reference files exist.**
If `build/reference-files` is missing or stale, call `./shamenv_do pull_reffiles` to fetch them.
Running tests without pulled reference files will produce failures.

```bash
# Step 1 — ensure reference files are pulled
cd build
test -d reference-files || ./shamenv_do pull_reffiles

# Step 2 — list devices for user selection
./shamenv_do ./shamrock --smi          # or ./shamrock_test --smi
```

Never truncate the output of `--smi` with `head` or similar — it contains device IDs needed to run tests.

Show the device table from the `--smi` output and **ask the user to select which device to use**. Do NOT pick a device yourself. **Prompt the user only once** and remember their choice for the rest of the session — reuse the same device for all subsequent test runs unless asked otherwise. Then run with the user-selected device ID:

```bash
./shamenv_do ./shamrock_test --sycl-cfg <user-chosen-id>:<user-chosen-id> --loglevel 1 --unittest
```

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
# First run ./shamenv_do ./shamrock --smi to list devices, ask user to pick, then:
./shamenv_do ./shamrock_test --sycl-cfg <user-chosen-id>:<user-chosen-id> --loglevel 1 --unittest
```
