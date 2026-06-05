VENV_DIR="$BUILD_DIR/.pyvenv"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi

source $VENV_DIR/bin/activate
