
def _calc_angular_momentum(data: 'SarracenDataFrame',
                           rbins: pd.Series,
                           origin: list,
                           unit_vector: bool):
    """
    Utility function to calculate angular momentum of the disc.

    Parameters
    ----------
    data: SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    rbins: Series
        The radial bin to which each particle belongs.
    origin: list
        The x, y and z centre point around which to compute radii.
    unit_vector: bool
        Whether to convert the angular momentum to unit vectors.
        Default is True.

    Returns
    -------
    Lx, Ly, Lz: Series
        The x, y and z components of the angular momentum per bin.
    """

    mass = _get_mass(data)

    x_data = data[data.xcol].to_numpy() - origin[0]
    y_data = data[data.ycol].to_numpy() - origin[1]
    z_data = data[data.zcol].to_numpy() - origin[2]

    Lx = y_data * data[data.vzcol] - z_data * data[data.vycol]
    Ly = z_data * data[data.vxcol] - x_data * data[data.vzcol]
    Lz = x_data * data[data.vycol] - y_data * data[data.vxcol]

    if isinstance(mass, float):
        Lx = (mass * Lx).groupby(rbins).sum()
        Ly = (mass * Ly).groupby(rbins).sum()
        Lz = (mass * Lz).groupby(rbins).sum()
    else:
        Lx = (data[data.mcol] * Lx).groupby(rbins).sum()
        Ly = (data[data.mcol] * Ly).groupby(rbins).sum()
        Lz = (data[data.mcol] * Lz).groupby(rbins).sum()

    if unit_vector:
        Lmag = 1.0 / np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)

        Lx = Lx * Lmag
        Ly = Ly * Lmag
        Lz = Lz * Lmag

    return Lx, Ly, Lz


def angular_momentum(data: 'SarracenDataFrame',
                     r_in: float = None,
                     r_out: float = None,
                     bins: int = 300,
                     geometry: str = 'cylindrical',
                     origin: list = None,
                     retbins: bool = False,
                     unit_vector: bool = True):
    """
    Calculates the angular momentum profile of the disc.

    The profile is computed by segmenting the particles into radial bins
    (rings) and summing the angular momentum of the particles within each
    bin.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    r_in : float, optional
        Inner radius of the disc. Defaults to the minimum r value.
    r_out : float, optional
        Outer radius of the disc. Defaults to the maximum r value.
    bins : int, optional
        Defines the number of equal-width bins in the range [r_in, r_out].
        Default is 300.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].
    retbins : bool, optional
        Whether to return the midpoints of the bins or not. Defaults to False.

    Returns
    -------
    array
        A NumPy array of length bins containing the angular momentum profile.
    array, optional
        The midpoint values of each bin. Only returned if *retbins=True*.

    Raises
    ------
    ValueError
        If the *geometry* is not *cylindrical* or *spherical*.
    """

    origin = _get_origin(origin)
    rbins, bin_edges = _bin_particles_by_radius(data, r_in, r_out, bins,
                                                geometry, origin)

    Lx, Ly, Lz = _calc_angular_momentum(data, rbins, origin, unit_vector)
    Lx, Ly, Lz = Lx.to_numpy(), Ly.to_numpy(), Lz.to_numpy()

    if retbins:
        return Lx, Ly, Lz, _get_bin_midpoints(bin_edges)
    else:
        return Lx, Ly, Lz