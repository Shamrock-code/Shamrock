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

### Build types / sanitizers

These are set during the `shamconfigure` step (after `source ./activate`):

- `ASAN`  — `-fsanitize=address`
- `UBSAN` — `-fsanitize=undefined`
- `TSAN`  — `-fsanitize=thread`
- `MSAN`  — `-fsanitize=memory`
- `COVERAGE` — `-fprofile-instr-generate -fcoverage-mapping` (forces
  SHAMROCK_USE_SHARED_LIB=Off)

## Testing

```bash
cd build
source ./activate
ctest          # runs the C++ test suite
```

Tests live in `src/tests/` alongside the library sources they exercise
(e.g. `src/tests/shambase/`, `src/tests/shammath/`). The project also has
Python tests but they are mainly run via CI.

## Code style & linting

- **Formatter**: `.clang-format` — LLVM style, 4-space indent, column limit
  **100**, C++17.
- **CI linter**: `.clang-tidy` — runs with `run-clang-tidy-20` (LLVM 20).
  Custom CheckOptions for naming: classes CamelCase, functions/lower_case,
  members lower_case.
- **Pre-commit**: clang-format (v22), cmake-format, ruff (Python), license/
  doxygen/pragma-once checks via local Python hooks.
- **Always run** `pre-commit run --all-files` before committing.

## Naming conventions (from .clang-tidy)

| Entity                         | Case       |
| ------ ------ ---------------- | ---------- |
| Class/Enum/Union               | CamelCase  |
| Function/Variable/Parameter/   | lower_case |
| Member                         | lower_case |

## Architecture overview

```text
src/
  shamrock/          core hydrodynamics solvers, mesh, Riemann
  shamphys/          physics models (EOS, reaction tables)
  shammodels/        higher-level models
  shambase/          base containers, math utils, I/O
  shammath/          math primitives (tensors, linear algebra)
  shamcomm/          MPI / SYCL comm layer
  shambackends/      SYCL backend implementations
  shamcmdopt/        CLI argument parsing
  shamsys/           system-level glue
  shamtree/          YAML tree / config parsing
  shamtest/          test helpers / macros
  shambindings/      pybind11 glue
  shampylib/         Python library entry point
  shamalsgs/         ALGOS (algorithms)
  pylib/             Python package root
  tests/             unit + integration tests
```

## CI highlights

- `main_workflow.yml` orchestrates all CI: source checks, docs, conda build,
  ASAN/UBSAN/TSAN.
- `on_pr.yml` gates merges on CI passing (respects `light-ci` label).
- `shamrock-acpp-clang-tidy.yml` — full clang-tidy on AdaptiveCpp + Ubuntu.
- `shamrock-acpp-clang-asan.yml`, `shamrock-acpp-clang-ubsan.yml` — sanitizer
  CI.
- `shamrock-acpp-phys-test.yml` — physics regression tests.

## Important constraints

- **SYCL backend is mandatory** — `--backend [SYCL|omp|hetero]` must be set
  during configure. Any major SYCL compiler works
  (AdaptiveCpp, DPC++, intel/llvm).
- **Submodules must be in sync** — CMake checks their commit hashes at configure
  time. Run `git pull --recurse-submodules`.
- **MacOS forces** `SHAMROCK_USE_SHARED_LIB=Off` due to known issues.
- **License headers** are checked at pre-commit — CeCILL boilerplate required on
  every source file.

## Files to avoid modifying unless explicitly asked

- `env/machine/*/setup-env.py` — machine-specific env configs.
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
cd build && source ./activate && shammake

# Run clang-tidy locally
CLANGTIDYBINARY=clang-tidy-20 run-clang-tidy-20 -p build/ \
  -use-color -config-file .clang-tidy

# Run pre-commit
pre-commit run --all-files

# Run tests
cd build && source ./activate && ctest

# Python bindings
cd build && source ./activate
python3 -c "import shampylib; shampylib.main()"
```
