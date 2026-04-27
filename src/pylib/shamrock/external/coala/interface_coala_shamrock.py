import numpy as np

from .generate_flux_intflux import compute_flux_coag_k0_kdv


def coala_source_term_k0(nbins, massgrid, rhodust, rhodust_eps, tensor_tabflux_coag, v_dust):
    """
    Function to compute the source for coagulation and fragmentation in continuity equation for piecewise constant approximation (see Lombart et al., 2021)
    Function for ballistic kernel with differential velocities dv
    Used to evaluate the source term, then hydro code applies time solver

    /!\ Only coagulation so far

    Parameters
    ----------
    nbins : scalar, type -> integer
       number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
       grid of masses given borders value of mass bins
    rhodust : 1D array (dim = nbins), type -> float
       dust density for each grain size
    rhodust_eps : scalar, type -> float
       threshold value for rhodust
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    v_dust : 1D array (dim = (nbins)), type -> float
       array of the dust velocities (could also be delta_v in monofluid since it is a delta)

    Returns
    -------
    S_coag : 1D array (dim = nbins), type -> float
        Source term for dust coagulation in continuity equation
        DG operator for piecewise constant approximation in each binls

    """

    # compute gij from rhodust for coala k=0
    gij = 0.0
    for j in range(nbins):
        if rhodust[j] > rhodust_eps:
            gij[j] = rhodust[j] / (massgrid[j + 1] - massgrid[j])

    # dv_ij = v_dust_j - v_dust_i
    dv = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            dv[i, j] = v_dust[j] - v_dust[i]

    # copmute flux for all dust bins
    flux = compute_flux_coag_k0_kdv(gij, tensor_tabflux_coag, dv)

    S_coag = np.zeros(nbins)
    S_coag[0] = -flux[0]
    S_coag[1:] = flux[:-1] - flux[1:]

    return S_coag
