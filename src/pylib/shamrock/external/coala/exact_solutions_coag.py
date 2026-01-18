import numpy as np
from mpmath import mp


def exact_sol_coag(kernel, x, tau):
    """
    Function to compute exact solution of the Smoluchowski equation for simple kernel (constant and additive)

    Parameters
    ----------
    kernel : scalar, type -> integer
       select the collisional kernel function
    x : scalar or 1D array, type -> float
       mass value to evaluate the solution
    tau : scalar, type -> float
       time to evaluate the solution


    Returns
    -------
    res : scalar or 1D array, type -> float
       exact solution evaluated at x and tau

    """
    match kernel:
        # solution kconst, K0 = 1
        # #g(x,0)=x exp(-x)
        case 0:
            if np.size(x) == 1:
                if tau == 0.0:
                    res = x * mp.exp(-x)
                else:
                    res = 4.0 * x / ((2.0 + tau) ** 2) * mp.exp(-(1.0 - tau / (2.0 + tau)) * x)

            else:
                if tau == 0:
                    res = [x[i] * mp.exp(-x[i]) for i in range(np.size(x))]
                else:
                    res = [
                        4.0 * x[i] / ((2.0 + tau) ** 2) * mp.exp(-(1.0 - tau / (2.0 + tau)) * x[i])
                        for i in range(np.size(x))
                    ]

            return res

        # solution kadd, K0 = 1
        # #g(x,0)=x exp(-x)
        case 1:
            if np.size(x) == 1:
                if tau == 0.0:
                    res = x * mp.exp(-x)
                else:
                    T = 1 - mp.exp(-tau)
                    res = ((1 - T) * mp.exp(-x * (T + 1)) * mp.besseli(1, 2 * x * mp.sqrt(T))) / (
                        mp.sqrt(T)
                    )

            else:
                if tau == 0:
                    res = [x[i] * mp.exp(-x[i]) for i in range(np.size(x))]
                else:
                    T = 1 - mp.exp(-tau)
                    res = [
                        ((1 - T) * mp.exp(-x[i] * (T + 1)) * mp.besseli(1, 2 * x[i] * mp.sqrt(T)))
                        / (mp.sqrt(T))
                        for i in range(np.size(x))
                    ]

            return res

        case _:
            raise ValueError("Need to choose a kernel with analytic solution.")
