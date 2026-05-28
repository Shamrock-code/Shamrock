# syntax=docker/dockerfile:1.7

FROM --platform=amd64 ghcr.io/shamrock-code/shamrock-ci:ubuntu22

ARG CLANGVERSION=18

# ----------------------------
# Install LLVM toolchain
# ----------------------------
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo apt update && \
    sudo ./llvm.sh ${CLANGVERSION} && \
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
    ccache

RUN pip3 install matplotlib numpy scipy h5py

# ----------------------------
# ccache configuration
# ----------------------------
ENV CCACHE_DIR=/tmp/ccache
ENV CCACHE_MAXSIZE=5G
ENV CCACHE_COMPRESS=1
ENV CCACHE_COMPRESSLEVEL=5

# sanity check (uses BuildKit cache)
RUN --mount=type=cache,target=/tmp/ccache \
    ccache --show-config && ccache --show-stats

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
    ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp && \
    cd build && \
    sed -i "s/-DBUILD_TEST=Yes /-DBUILD_TEST=Yes -DCXX_FLAG_ARCH_NATIVE=off /" activate && \
    cat activate && \
    pip3 install -v .

# ----------------------------
# Final environment
# ----------------------------
WORKDIR /work
