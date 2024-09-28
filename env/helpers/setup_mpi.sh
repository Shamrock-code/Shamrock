#!/bin/bash

if [ -z ${UCX_INSTALL_PATH+x} ]; then echo "UCX_INSTALL_PATH is unset"; return; fi
if [ -z ${OMPI_INSTALL_PATH+x} ]; then echo "OMPI_INSTALL_PATH is unset"; return; fi
if [ -z ${OMPI_SOURCE_DIR+x} ]; then echo "OMPI_SOURCE_DIR is unset"; return; fi
if [ -z ${CUDA_PATH+x} ]; then echo "CUDA_PATH is unset"; return; fi

if [ -z ${UCX_URL+x} ]; then echo "UCX_URL is unset"; return; fi
if [ -z ${OMPI_URL+x} ]; then echo "OMPI_URL is unset"; return; fi

export PATH=$OMPI_INSTALL_PATH/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$OMPI_INSTALL_PATH/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=$UCX_INSTALL_PATH/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$UCX_INSTALL_PATH/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir -p $OMPI_SOURCE_DIR

function buildinstall_ucx {

    ### UCX section
    wget -O ucx.tar.gz $UCX_URL
    tar -xf ucx.tar.gz
    rm ucx.tar.gz
    mv ucx $OMPI_SOURCE_DIR/ucx

    cd $OMPI_SOURCE_DIR/ucx

    ./configure --prefix=$UCX_INSTALL_PATH --with-cuda=$CUDA_PATH --without-go

    make -j64
    make install
}

function buildinstall_ompi {

    ### Ompi section
    wget -O openmpi.tar.gz $OMPI_URL
    tar -xf openmpi.tar.gz
    rm openmpi.tar.gz
    mv openmpi $OMPI_SOURCE_DIR/openmpi

    cd $OMPI_SOURCE_DIR/openmpi

    ./configure --prefix=$OMPI_INSTALL_PATH --with-cuda=$CUDA_PATH --with-ucx=$UCX_INSTALL_PATH

    make -j64
    make install
}

if [ ! -f "$UCX_INSTALL_PATH/ucx_info" ]; then
    echo " ------ Building UCX ------ "
    buildinstall_ucx
fi

if [ ! -f "$OMPI_INSTALL_PATH/ompi_info" ]; then
    echo " ------ Building OpenMPI ------ "
    buildinstall_ompi
fi
