import numpy as np

# from numba import njit


# @njit
def kconst(K0, u, v):
    """
    Function to compute constant kernel

    Parameters
    ----------
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)


    Returns
    -------
    res : scalar, type -> float
       evaluate constant kernel

    """

    res = K0

    return res


# @njit
def kadd(K0, u, v):
    """
    Function to compute additive kernel

    Parameters
    ----------
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)


    Returns
    -------
    res : scalar, type -> float
       evaluate additive kernel at u and v

    """

    res = K0 * (u + v)
    return res


# @njit
def kdv(K0, u, v):
    """
    Function to compute the cross-section term in the ballistic kernel K = sigma * dv

    Parameters
    ----------
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)


    Returns
    -------
    res : scalar, type -> float
       evaluate cross-section term of the balistic kernel at u and v

    """

    # cross-section is calculated from mass variables
    res = K0 * (u ** (2.0 / 3.0) + 2.0 * u ** (1.0 / 3.0) * v ** (1.0 / 3.0) + v ** (2.0 / 3.0))

    return res


# @njit
def k_Br(K0, u, v):
    """
    Function to compute collision kernel from Brownian motion, K = sigma * dv

    Parameters
    ----------
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)


    Returns
    -------
    res : scalar, type -> float
       Brownian motion collision kernel

    """

    res = (
        K0
        * (u ** (2.0 / 3.0) + 2.0 * u ** (1.0 / 3.0) * v ** (1.0 / 3.0) + v ** (2.0 / 3.0))
        * np.sqrt(1.0 / u + 1.0 / v)
    )

    return res


def func_kernel(kernel, K0, u, v):
    """
    Function to compute kernels at u and v

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)


    Returns
    -------
    res : scalar, type -> float
       evaluate kernel at u and v

    """

    match kernel:
        case 0:
            res = kconst(K0, u, v)
        case 1:
            res = kadd(K0, u, v)
        case 2:
            res = k_Br(K0, u, v)
        case 3:
            res = kdv(K0, u, v)
        case _:
            return "Need to choose a kernel in the list."

    return res


# @njit
def func_kernel_numba(kernel, K0, u, v):
    """
    Function to compute kernels at u and v

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    K0 : scalar, type -> float
       constant value of the kernel function (used to adapt to code unit)
    u : scalar, type -> float
       mass variable (colliding grain of mass u)
    v : scalar, type -> float
       mass variable (colliding grain of mass v)


    Returns
    -------
    res : scalar, type -> float
       evaluate kernel at u and v

    """

    if kernel == 0:
        res = kconst(K0, u, v)
    elif kernel == 1:
        res = kadd(K0, u, v)
    elif kernel == 2:
        res = k_Br(K0, u, v)
    elif kernel == 3:
        res = kdv(K0, u, v)
    else:
        # Return special value for unsupported kernel; cannot return string in njit
        res = -1.0
    return res
