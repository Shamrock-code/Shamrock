# syntax=docker/dockerfile:1.7

FROM --platform=amd64 ghcr.io/shamrock-code/shamrock-ci:ubuntu22

ARG CLANGVERSION=18

# ----------------------------
# Install LLVM toolchain
# ----------------------------
# apt.llvm.org is served via Fastly, which advertises IPv6 addresses that
# GitHub-hosted runners often can't route reliably, causing intermittent
# "Network is unreachable" failures in apt/wget (including GPG key fetch).
# Force IPv4 for apt and wget, and retry llvm.sh to absorb transient
# Fastly edge failures. (Claude generated comment)
RUN sudo sh -c 'echo "Acquire::ForceIPv4 \"true\";" > /etc/apt/apt.conf.d/99force-ipv4' && \
    sudo apt update && \
    wget -4 https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    for i in 1 2 3 4 5; do \
        sudo ./llvm.sh ${CLANGVERSION} && break; \
        echo "llvm.sh failed (attempt $i), retrying..."; \
        sleep 15; \
    done && \
    sudo apt install -y \
    libclang-${CLANGVERSION}-dev \
    clang-tools-${CLANGVERSION} \
    libomp-${CLANGVERSION}-dev && \
    rm llvm.sh

# ----------------------------
# Base dependencies
# ----------------------------
RUN sudo apt install -y \
    gfortran \
    ffmpeg \
    ccache \
    graphviz

RUN pip3 install matplotlib numpy scipy h5py graphviz

# ----------------------------
# ccache configuration
# ----------------------------
ENV CCACHE_DIR=/tmp/ccache

# sanity check (uses BuildKit cache)
RUN --mount=type=cache,target=/tmp/ccache \
    ccache --show-config && ccache --show-stats && \
    ccache -o compiler_check='%compiler% -dumpmachine; %compiler% -dumpversion'

# ----------------------------
# Workspace
# ----------------------------
WORKDIR /opt/Shamrock_src

ADD repo.tar.gz .

RUN ls -la

# ----------------------------
# Build
# ----------------------------
RUN --mount=type=cache,target=/tmp/ccache \
    ccache --show-stats && \
    ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp && \
    cd build && \
    sed -i "s/-DBUILD_TEST=Yes /-DBUILD_TEST=Yes -DCXX_FLAG_ARCH_NATIVE=off /" activate && \
    cat activate && \
    pip3 install -v . && \
    ccache --show-stats

# ----------------------------
# Final environment
# ----------------------------
WORKDIR /work
