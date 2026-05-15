#!/bin/bash

if [ -z ${SPHINX_VENV_DIR+x} ]; then
    echo "error: SPHINX_VENV_DIR is unset"
    return
fi

function setup_sphinx_env {

    if [ -z ${SPHINX_PYTHON_EXECUTABLE+x} ]; then
        echo "error: SPHINX_PYTHON_EXECUTABLE is unset, using default python executable"
        echo "Running : python3 -c \"import sys; print(sys.executable)\""
        PYSPHINX=$(python3 -c "import sys; print(sys.executable)")
    else
        PYSPHINX=${SPHINX_PYTHON_EXECUTABLE}
    fi

    echo "Using Python executable: ${PYSPHINX}"

    if [ ! -f "$SPHINX_VENV_DIR/bin/activate" ]; then
        echo "No python venv were configured for the doc, creating it in $SPHINX_VENV_DIR"
        $PYSPHINX -m venv $SPHINX_VENV_DIR
    fi

    (
        source $SPHINX_VENV_DIR/bin/activate &&
            pip install -U \
                matplotlib \
                numpy \
                scipy \
                sphinx \
                pydata-sphinx-theme \
                sphinx-gallery \
                memory-profiler \
                sphinx-copybutton \
                sphinx_design \
                imageio \
                sympy \
                myst-parser \
                mystmd
    )

}

if [ -z ${BUILD_DIR+x} ]; then
    echo "error: BUILD_DIR is unset"
    return
fi

export SPHINX_SHAMROCK_INSTALL_DIR=$BUILD_DIR/shaminstall
export SPHINX_SHAMROCK_PYTHON_DIR=$BUILD_DIR/pysham

function local_shamrock_install {
    echo "-- Installing Shamrock a local folder environment --"

    # Store the current cmake variables CMAKE_INSTALL_PYTHONDIR &
    # CMAKE_INSTALL_PREFIX to restore them later
    local OLD_CMAKE_INSTALL_PYTHONDIR=$(cmake -L -N $BUILD_DIR 2>/dev/null | grep CMAKE_INSTALL_PYTHONDIR | cut -d'=' -f2)
    local OLD_CMAKE_INSTALL_PREFIX=$(cmake -L -N $BUILD_DIR 2>/dev/null | grep CMAKE_INSTALL_PREFIX | cut -d'=' -f2)

    # Change the install folders for sphinx
    cmake $BUILD_DIR -DCMAKE_INSTALL_PYTHONDIR=$SPHINX_SHAMROCK_PYTHON_DIR -DCMAKE_INSTALL_PREFIX=$SPHINX_SHAMROCK_INSTALL_DIR

    # Install
    shammake install

    # Restore the original cmake variables
    cmake $BUILD_DIR -DCMAKE_INSTALL_PYTHONDIR=$OLD_CMAKE_INSTALL_PYTHONDIR -DCMAKE_INSTALL_PREFIX=$OLD_CMAKE_INSTALL_PREFIX
}

function _run_sphinx_doc_gen {
    local description="$1"
    local script="$2"

    echo "-- Generating Sphinx documentation ($description) --"

    echo "-- Setting up sphinx environment --"
    setup_sphinx_env

    echo "-- Installing Shamrock in a local folder environment --"
    local_shamrock_install

    (
        echo "-- Activating sphinx environment --"
        source $SPHINX_VENV_DIR/bin/activate

        cd $SHAMROCK_DIR/doc/sphinx

        echo "-- Generating Sphinx documentation ($description) --"
        export PYTHONPATH=$SPHINX_SHAMROCK_PYTHON_DIR:$PYTHONPATH
        case "$script" in
        single)
            bash gen_sphinx_doc_single_example.sh $3
            ;;
        full)
            bash gen_sphinx_doc.sh
            ;;
        *)
            echo "error: unknown script type '$script', expected 'single' or 'full'"
            return 1
            ;;
        esac
    )
}

function generate_sphinx_doc_no_examples {
    _run_sphinx_doc_gen "without examples" single do_not_run_annything_dammit
}

function generate_sphinx_doc_with_examples {
    _run_sphinx_doc_gen "with examples" full
}

function generate_sphinx_doc_single_example {
    _run_sphinx_doc_gen "single example" single $1
}
