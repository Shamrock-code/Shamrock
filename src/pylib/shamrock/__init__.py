try:
    # try to import from the global namespace (works if embedded python interpreter is used)
    from pyshamrock import *

    SHAM_IMPORT_MODE = "global"
except ImportError:
    # then it is a library mode, we import from the local namespace
    from .pyshamrock import *

    SHAM_IMPORT_MODE = "local"

# print(f"shamrock imported from {__file__}")
# print(f"import log: {SHAM_IMPORT_MODE}")
