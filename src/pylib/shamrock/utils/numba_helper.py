try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def maybe_njit(fn):
    if HAS_NUMBA:
        return njit(fn)
    return fn
