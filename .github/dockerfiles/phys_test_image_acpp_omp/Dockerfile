FROM --platform=amd64 ghcr.io/shamrock-code/shamrock-ci:ubuntu20

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh ${CLANGVERSION} && \
    sudo apt install -y libclang-${CLANGVERSION}-dev clang-tools-${CLANGVERSION} libomp-${CLANGVERSION}-dev && \
    rm llvm.sh

RUN pip install matplotlib numpy scipy

WORKDIR /opt/Shamrock_src

COPY . .

RUN ls -la

RUN ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp && \
    cd build && \
    pip3 install -v --break-system-packages .

WORKDIR /work
