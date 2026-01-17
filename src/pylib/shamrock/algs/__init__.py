"""
Shamrock algorithmic library.
"""

try:
    # try to import from the global namespace (works if embedded python interpreter is used)
    from pyshamrock.algs import *
except ImportError:
    # then it is a library mode, we import from the local namespace
    from ..pyshamrock.algs import *


# explicitly re-export public API
__all__ = [name for name in globals() if not name.startswith("_")]

# Sphinx uses obj.__module__ to decide where something belongs.
for name in __all__:
    try:
        globals()[name].__module__ = __name__
    except (AttributeError, TypeError):
        # Some C-extension objects or builtins don't allow rebinding __module__
        pass

# print(f"shamrock.algs.__all__: {__all__}")
