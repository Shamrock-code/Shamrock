VENV_DIR="$BUILD_DIR/.pyvenv"

if [ ! -d "$VENV_DIR" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "No python or python3 found" >&2
        exit 1
    fi

    "$PYTHON" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi

source $VENV_DIR/bin/activate
