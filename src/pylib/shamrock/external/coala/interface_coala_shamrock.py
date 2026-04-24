from .solver_DG import *

def coala_source_term_k0(nbins,massgrid, rhodust, tensor_tabflux_coag, dv):
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
    tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)), type -> float
       array to evaluate coagulation flux
    dv : 2D array (dim = (nbins,nbins)), type -> float
       array of the differential velocity between grains


    Returns
    -------
    S_coag : 1D array (dim = nbins), type -> float
        Source term for dust coagulation in continuity equation
        DG operator for piecewise constant approximation in each binls       

    """

    #compute gij from rhodust for coala k=0
    gij = rhodust/(massgrid[1:] - massgrid[0:nbins])

    #copmute flux for all dust bins
    flux = compute_flux_coag_k0_kdv(gij, tensor_tabflux_coag, dv)

    S_coag = np.zeros(len(rhodust))
    S_coag[0] = -flux[0]
    S_coag[1:] = flux[:-1] - flux[1:]

    return S_coag