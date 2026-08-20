#!/bin/bash
set -euo pipefail

# Only run this setup in Claude Code on the web / remote sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# CLAUDE_PROJECT_DIR is unset when the environment is started from a
# manually configured directory (e.g. pasted into the environment's setup
# script box) rather than the normal session bootstrap; in that case we're
# already in the right directory, so just skip the cd.
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  cd "$CLAUDE_PROJECT_DIR"
fi

# --- System dependencies -----------------------------------------------
# AdaptiveCpp (SYCL) needs Boost.context/fiber and an LLVM install; Shamrock
# needs an MPI implementation; pre-commit is used for linting; clang-tidy-20/
# clangd-20 back dev-tooling. A single LLVM 20 toolchain backs both the
# AdaptiveCpp build and dev tooling (AdaptiveCpp supports up to LLVM 20 per
# its CMakeLists.txt) — 20 is the newest available directly from Ubuntu
# noble's own repos; apt.llvm.org (which would offer newer releases closer
# to the clang-format v22.1.8 the `pre-commit` config pins to, matching the
# `.clangd` file's `>= clangd-21`/`>= clangd-22` comments) is blocked by
# this environment's network policy.
NEEDED_PKGS="libboost-context-dev libboost-fiber-dev llvm-20-dev libclang-20-dev libomp-20-dev libopenmpi-dev openmpi-bin pre-commit clang-20 clangd-20 clang-tidy-20"
MISSING_PKGS=""
for pkg in $NEEDED_PKGS; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS $pkg"
  fi
done
if [ -n "$MISSING_PKGS" ]; then
  # Some base images ship extra apt sources (e.g. deadsnakes/ondrej PPAs)
  # that this environment's network policy blocks; that makes `apt-get
  # update` exit non-zero even though the archives we actually need
  # (Ubuntu main/universe/security) refresh fine. Don't let that abort us.
  apt-get update || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y $MISSING_PKGS
fi

# clang-20/clangd-20 only install versioned /usr/bin/*-20 binaries, not
# plain names on PATH; give clangd the unversioned name so it's invocable
# directly (e.g. `clangd --check=<file>`).
if ! command -v clangd >/dev/null 2>&1 && [ -x /usr/bin/clangd-20 ]; then
  ln -sf /usr/bin/clangd-20 /usr/local/bin/clangd
fi

# pre-commit's isolated venvs pick up Debian's patched sysconfig scheme,
# which expects a distutils "install_layout" attribute that setuptools'
# vendored (local) distutils no longer provides. Forcing stdlib distutils
# avoids the AttributeError when hook environments are built.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export SETUPTOOLS_USE_DISTUTILS=stdlib' >> "$CLAUDE_ENV_FILE"
fi
export SETUPTOOLS_USE_DISTUTILS=stdlib

# --- Submodules ----------------------------------------------------------
git submodule update --init --recursive --jobs "$(nproc)"

# --- Build environment -----------------------------------------------
# CPU-only container: use AdaptiveCpp's OpenMP backend (no GPU present).
# Deliberately stop here: `shamenv_do shamconfigure` builds AdaptiveCpp
# from source on its first invocation (a few minutes), so it's left for
# whenever a build/test is actually needed rather than blocking every
# session start. That first `shamconfigure`/`shammake` call will pay the
# one-time cost inline; every session after that reuses the cached build.
if [ ! -f build/shamenv_do ]; then
  ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp
fi
