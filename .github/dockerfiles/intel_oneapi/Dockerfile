FROM --platform=amd64 intel/oneapi

RUN apt-get update && \
    apt-get install -y python3-dev ninja-build python3-full python3-pip && \
    apt-get clean

WORKDIR /opt/Shamrock_src

ADD repo.tar.gz .

RUN ls -la

RUN ./env/new-env --machine docker.intel_oneapi --builddir build_intel_docker -- && \
    cd build_intel_docker && \
    pip3 install -v --break-system-packages .

# Installed through apt rather than pip: the base image already ships some
# Python packages through apt (e.g. typing_extensions), which pip cannot
# uninstall (no RECORD file) when a dependency needs a newer version.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        jupyter jupyterlab jupyter-notebook jupyter-console jupyter-nbconvert \
        python3-ipykernel python3-ipywidgets \
        python3-matplotlib python3-numpy python3-scipy && \
    apt-get clean
EXPOSE 8888

WORKDIR /work
