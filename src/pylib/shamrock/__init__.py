try:
    # try to import from the global namespace (works if embedded python interpreter is used)
    from pyshamrock import *

    IMPORT_LOG = "global"
except ImportError:
    # then it is a library mode, we import from the local namespace
    from .pyshamrock import *

    IMPORT_LOG = "local"

print(f"pyshamrock imported from {__file__}")
print(f"import log: {IMPORT_LOG}")
