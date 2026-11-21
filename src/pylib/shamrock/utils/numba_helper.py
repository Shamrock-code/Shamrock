try:
    from numba import njit
    from numba.core.dispatcher import Dispatcher

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def maybe_njit(fn):
    """
    Apply numba.njit when numba is installed, and return fn unchanged otherwise.

    Functions that are already jitted are returned as is, since numba refuses to
    jit a dispatcher a second time (e.g. a profile from shamrock.utils.disc_setup
    handed to one of the analysis plots).
    """
    if HAS_NUMBA and not isinstance(fn, Dispatcher):
        return njit(fn)
    return fn
