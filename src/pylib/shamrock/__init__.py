"""
Main shamrock module.
"""

try:
    # try to import from the global namespace (works if embedded python interpreter is used)
    from pyshamrock import *

    SHAM_IMPORT_MODE = "global"
except ImportError:
    # then it is a library mode, we import from the local namespace
    from .pyshamrock import *

    SHAM_IMPORT_MODE = "local"

# explicitly re-export public API
__all__ = [name for name in globals() if not name.startswith("_") and not name == "pyshamrock"]

# Sphinx uses obj.__module__ to decide where something belongs.
for name in __all__:
    try:
        globals()[name].__module__ = __name__
    except (AttributeError, TypeError):
        # Some C-extension objects or builtins don't allow rebinding __module__
        pass

from . import utils

# print(f"shamrock.__all__: {__all__}")
# print(f"shamrock imported from {__file__}")
# print(f"import log: {SHAM_IMPORT_MODE}")
