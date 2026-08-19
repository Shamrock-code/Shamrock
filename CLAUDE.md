@AGENTS.md

## Claude Code on the web: container setup

This container has no GPU, so AdaptiveCpp is built from source targeting
the OpenMP backend. Automated by `.claude/hooks/session-start.sh`; the
condensed steps it runs:

```bash
# System packages (Boost.context/fiber + LLVM for AdaptiveCpp, OpenMPI, pre-commit)
apt-get install -y libboost-context-dev libboost-fiber-dev llvm-18-dev \
  libclang-18-dev libomp-18-dev libopenmpi-dev openmpi-bin pre-commit

# Submodules
git submodule update --init --recursive

# Env + configure (builds the AdaptiveCpp compiler on first run)
./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp
cd build && ./shamenv_do shamconfigure
```

pre-commit hook venvs also need `SETUPTOOLS_USE_DISTUTILS=stdlib` exported —
Debian's patched sysconfig scheme otherwise breaks setuptools' vendored
distutils with `AttributeError: install_layout`.
