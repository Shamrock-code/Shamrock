try:
    from .pyshamrock import *
except ImportError:
    try:
        from shamrock import *
    except ImportError:
        from pyshamrock import *
    