FROM --platform=amd64 intel/oneapi

RUN apt-get update && \
    apt-get install -y python3-dev ninja-build python3-full python3-pip && \
    apt-get clean

WORKDIR /opt/Shamrock_src

COPY . .

RUN ls -la

RUN ./env/new-env --machine docker.intel_oneapi --builddir build_intel_docker -- && \
    cd build_intel_docker && \
    pip3 install -v --break-system-packages .

RUN pip3 install -v --break-system-packages jupyter
EXPOSE 8888

WORKDIR /work
