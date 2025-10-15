import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

##### ==============GRID =============

# Grid parameters.
nx = 8  # 101                  # number of points in the x direction
ny = 8  # 101                  # number of points in the y direction
xmin, xmax = 0.0, 1.0  # limits in the x direction
ymin, ymax = -0.5, 0.5  # limits in the y direction
lx = xmax - xmin  # domain length in the x direction
ly = ymax - ymin  # domain length in the y direction
dx = lx / (nx - 1)  # grid spacing in the x direction
dy = ly / (ny - 1)  # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing="ij")


def rhs_2d(X, Y, A=1e6, Lx=1.0, Ly=1.0):
    """
    Computes the source term b(x, y) = A * sin(2πx/Lx) * sin(2πy/Ly)

    Parameters:
    - X, Y: 2D meshgrid arrays
    - A: Amplitude (default 1.0)
    - Lx: Domain length in x-direction (default 1.0)
    - Ly: Domain length in y-direction (default 1.0)

    Returns:
    - b: 2D array with the same shape as X and Y, representing the source term
    """
    return 4 * np.pi * np.sin(2 * np.pi * (X / Lx)) * np.cos(12 * np.pi * (Y / Ly))

    # return A * (-4*np.pi*np.sin(0.5 * np.pi * (X / Lx))+ (20*np.pi)* np.cos(1000 * np.pi * (Y / Ly)))


# # Create the source term.
b = rhs_2d(X, Y)


###================ def AV
def A(v, dx, dy):
    """
    Computes the action of (-) the Poisson operator on any
    vector v_{ij} for the interior grid nodes

    Parameters
    ----------
    v : numpy.ndarray
        input vector
    dx : float
         grid spacing in the x direction
    dy : float
        grid spacing in the y direction


    Returns
    -------
    Av : numpy.ndarray
        action of A on v
    """

    Av = -(
        (v[:-2, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[2:, 1:-1]) / dx**2
        + (v[1:-1, :-2] - 2.0 * v[1:-1, 1:-1] + v[1:-1, 2:]) / dy**2
    )

    return Av


def l2_diff(f1, f2):
    """
    Computes the l2-norm of the difference
    between a function f1 and a function f2

    Parameters
    ----------
    f1 : array of floats
        function 1
    f2 : array of floats
        function 2

    Returns
    -------
    diff : float
        The l2-norm of the difference.
    """
    l2_diff = np.sqrt(np.sum((f1 - f2) ** 2)) / f1.size

    return l2_diff


######===========
# Initial guess
p0 = np.zeros((nx, ny))

# Place holders for the residual r and A(d)
r = np.zeros((nx, ny))
Ad = np.zeros((nx, ny))

# Create the source term.
b = rhs_2d(X, Y)


####============

tolerance = 1e-10
max_it = 1000


it = 0  # iteration counter
diff = 1.0
tol_hist_jac = []

p = p0.copy()

# Initial residual r0 and initial search direction d0
r[1:-1, 1:-1] = -b[1:-1, 1:-1] - A(p, dx, dy)
d = r.copy()

while diff > tolerance:
    if it > max_it:
        print(
            "\nSolution did not converged within the maximum"
            " number of iterations"
            f"\nLast l2_diff was: {diff:.5e}"
        )
        break

    # Laplacian of the search direction.
    Ad[1:-1, 1:-1] = A(d, dx, dy)
    # Magnitude of jump.
    alpha = np.sum(r * r) / np.sum(d * Ad)
    print(f"alpha = {alpha}")
    # Iterated solution
    pnew = p + alpha * d
    # Intermediate computation
    beta_denom = np.sum(r * r)
    # Update the residual.
    r = r - alpha * Ad
    print(f" r2 = {np.dot(r,r)}\n")

    # Compute beta
    beta = np.sum(r * r) / beta_denom
    # Update the search direction.
    d = r + beta * d

    diff = l2_diff(pnew, p)
    print(f"diff = {diff}\n")
    # tol_hist_jac.append(diff)

    # Get ready for next iteration
    it += 1
    np.copyto(p, pnew)


else:
    print(f"\nThe solution converged after {it} iterations")
