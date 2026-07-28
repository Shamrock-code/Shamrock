import numpy as np

from .StandardPlotHelper import StandardPlotHelper

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def SliceByPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    """Render a slice of the By magnetic field component.

    The MHD solver stores ``B/rho`` (a 3-vector) so By is recovered as ``(B/rho)_y * rho``.
    """

    def compute_By_slice(helper):
        def custom_getter(size: int, dic_out: dict) -> np.array:
            B_over_rho_y = dic_out["B/rho"][:, 1]
            pmass = model.get_particle_mass()
            hfact = model.get_hfact()
            rho = pmass * (hfact / dic_out["hpart"]) ** 3
            return B_over_rho_y * rho

        arr_By = helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=custom_getter,
        )

        return arr_By

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_By_slice,
    )


def SliceBthetaPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    """Render a slice of the azimuthal magnetic field component B_theta.

    the azimuthal projection is ``(-y * Bx + x * By) / r``.
    """

    def compute_Btheta_slice(helper):
        def internal(
            size: int, x: np.array, y: np.array, Bx: np.array, By: np.array, rho: np.array
        ) -> np.array:
            r_safe = np.sqrt(x**2 + y**2) + 1e-9
            B_over_rho_theta = (-y * Bx + x * By) / r_safe
            return B_over_rho_theta * rho  # recover B_theta = (B/rho)_theta * rho

        if _HAS_NUMBA:
            internal = njit(internal)

        def custom_getter(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
                dic_out["B/rho"][:, 0],
                dic_out["B/rho"][:, 1],
                dic_out["hfact"],
            )

        arr_Btheta = helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=custom_getter,
        )

        return arr_Btheta

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_Btheta_slice,
    )


def SliceBVerticalShearGradient(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    """Render a slice of the vertical shear gradient of the azimuthal magnetic field,
    ``d(B_theta)/dz``

    ``B_theta`` is recovered from the stored ``B/rho`` field as
    ``(B/rho)_theta * rho``.
    """

    def compute_B_vertical_shear_gradient(helper):
        def internal(
            size: int, x: np.array, y: np.array, Bx: np.array, By: np.array, rho: np.array
        ) -> np.array:
            r_safe = np.sqrt(x**2 + y**2) + 1e-9
            B_over_rho_theta = (-y * Bx + x * By) / r_safe
            return B_over_rho_theta * rho  # recover B_theta = (B/rho)_theta * rho

        if _HAS_NUMBA:
            internal = njit(internal)

        def custom_getter(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
                dic_out["B/rho"][:, 0],
                dic_out["B/rho"][:, 1],
                dic_out["hfact"],
            )

        arr_Btheta = helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=custom_getter,
        )

        extent = helper.get_extent()
        dy = (extent[3] - extent[2]) / helper.ny

        B_vertical_shear_gradient = np.gradient(arr_Btheta, dy, axis=0)

        return B_vertical_shear_gradient

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_B_vertical_shear_gradient,
    )
