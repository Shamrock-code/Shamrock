#!/bin/bash
set -euo pipefail

# Hard deadline for this whole hook, in seconds from script start (bash's
# $SECONDS). The cloud environment snapshot is taken after this hook finishes,
# but only when setup stays under roughly five minutes; 4m30 keeps a margin.
HOOK_DEADLINE=270

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
#
# ccache: the debian-generic.acpp env script passes
# -DCMAKE_CXX_COMPILER_LAUNCHER=ccache to both the AdaptiveCpp build and
# `shamconfigure` whenever `ccache` is on PATH (checked each time
# `shamenv_do` sources the env), so installing it is all that's needed to
# enable it.
NEEDED_PKGS="libboost-context-dev libboost-fiber-dev llvm-20-dev libclang-20-dev libomp-20-dev libopenmpi-dev openmpi-bin pre-commit clang-20 clangd-20 clang-tidy-20 ccache"
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

# pre-commit builds hook environments with the container's python3 (3.13),
# which has no stdlib distutils, so setuptools must use its vendored (local)
# copy. Export it explicitly to override a leftover
# SETUPTOOLS_USE_DISTUTILS=stdlib, which fails with
# "No module named 'distutils'".
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export SETUPTOOLS_USE_DISTUTILS=local' >> "$CLAUDE_ENV_FILE"
fi
export SETUPTOOLS_USE_DISTUTILS=local

# --- Submodules ----------------------------------------------------------
git submodule update --init --recursive --jobs "$(nproc)"

# --- Build environment -----------------------------------------------
# CPU-only container: use AdaptiveCpp's OpenMP backend (no GPU present).
if [ ! -f build/shamenv_do ]; then
  ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp
fi

# --- Time-boxed pre-build ----------------------------------------------
# The environment cache snapshots the disk right after this hook, so whatever
# gets built here carries over to every session started from that snapshot.
# Each step gets what is left of HOOK_DEADLINE; a step that is cut short is
# picked up again by the next `shamconfigure`/`shammake` (ninja reruns
# unfinished steps, ccache keeps finished ones).
#
# Only on a cold start, i.e. when the repo was cloned during this boot: the
# cache-building run, or a session that starts while the cache rebuilds. A
# session restored from the snapshot (repo cloned before this boot) skips it
# even if the snapshot's pre-build is incomplete: only the cache-building run
# is snapshotted, so redoing it there would just block that session's start
# for minutes; whatever is missing gets built when it is first needed.
run_until_deadline() {
  local left=$((HOOK_DEADLINE - SECONDS - 5)) # 5 s for timeout's -k grace
  if [ "$left" -le 0 ]; then
    return 124
  fi
  timeout -s INT -k 5 "$left" "$@"
}

boot_time=$(awk '/^btime/ {print $2}' /proc/stat)
clone_time=$(stat -c %W .git) # birth time; 0 if unknown, i.e. treated as warm
if [ "$clone_time" -ge "$boot_time" ]; then
  # Sourcing the env builds AdaptiveCpp on first use (~2 min cold).
  acpp_rc=0
  run_until_deadline build/shamenv_do true || acpp_rc=$?
  if [ "$acpp_rc" -ne 0 ]; then
    # The env script treats AdaptiveCpp as built once bin/acpp exists, which a
    # stop during `make install` could leave behind; drop the install so the
    # next session resumes the (kept) build instead of using a partial one.
    rm -rf build/.env/acpp-installdir
    echo "pre-build: AdaptiveCpp not finished (rc=$acpp_rc) at ${SECONDS}s"
  else
    conf_rc=0
    run_until_deadline build/shamenv_do shamconfigure || conf_rc=$?
    make_rc=skipped
    if [ "$conf_rc" -eq 0 ] && [ -f build/build.ninja ]; then
      make_rc=0
      run_until_deadline build/shamenv_do shammake || make_rc=$?
    fi
    echo "pre-build: shamconfigure rc=$conf_rc, shammake rc=$make_rc at ${SECONDS}s"
  fi
fi
