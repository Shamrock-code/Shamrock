FROM --platform=amd64 ghcr.io/shamrock-code/shamrock-ci:ubuntu20

ARG CLANGVERSION=18

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh ${CLANGVERSION} && \
    sudo apt install -y libclang-${CLANGVERSION}-dev clang-tools-${CLANGVERSION} libomp-${CLANGVERSION}-dev && \
    rm llvm.sh

RUN pip install matplotlib numpy scipy h5py

WORKDIR /opt/Shamrock_src

COPY . .

RUN ls -la

RUN ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp && \
    cd build && \
    sed -i "s/-DBUILD_TEST=Yes /-DBUILD_TEST=Yes -DCXX_FLAG_ARCH_NATIVE=off /" activate && \
    cat activate && \
    pip3 install -v .

WORKDIR /work
