try:
    # try to import from the global namespace (works if embedded python interpreter is used)
    from pyshamrock.sys import *
except ImportError:
    # then it is a library mode, we import from the local namespace
    from .shamrock.sys import *
