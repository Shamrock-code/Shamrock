import numpy as np

import shamrock.sys

from .StandardPlotHelper import StandardPlotHelper

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def SliceVzPlot(
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
    def compute_v_z_slice(helper):
        def keep_only_v_z(arr_v):
            return arr_v[:, :, 2]

        arr_v = helper.slice_render(
            "vxyz", "f64_3", do_normalization, min_normalization, keep_only_v_z
        )

        return arr_v

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
        compute_function=compute_v_z_slice,
    )


def ColumnAverageVzPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    min_normalization=1e-9,
):
    def compute_v_z_slice(helper):
        def custom_getter(size: int, dic_out: dict) -> np.array:
            return dic_out["vxyz"][:, 2]

        arr_v = helper.column_average_render(
            "custom", "f64", min_normalization, custom_getter=custom_getter
        )

        return arr_v

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
        compute_function=compute_v_z_slice,
    )


def SliceDiffVthetaProfile(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    velocity_profile,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_diff_vtheta_profile(helper):
        if _HAS_NUMBA:
            if shamrock.sys.world_rank() == 0:
                print("Using numba for velocity profile in SliceDiffVthetaProfile")

        if _HAS_NUMBA:
            vel_profile_jit = njit(velocity_profile)
        else:
            vel_profile_jit = np.vectorize(velocity_profile)

        def internal(
            size: int, x: np.array, y: np.array, vx: np.array, vy: np.array, vz: np.array
        ) -> np.array:
            r = np.sqrt(x**2 + y**2)
            r_safe = r + 1e-9
            v_theta = (-y * vx + x * vy) / r_safe
            v_relative = v_theta - vel_profile_jit(r)
            return v_relative

        if _HAS_NUMBA:
            internal = njit(internal)

        def custom_getter(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
                dic_out["vxyz"][:, 0],
                dic_out["vxyz"][:, 1],
                dic_out["vxyz"][:, 2],
            )

        arr_v = helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=custom_getter,
        )

        return arr_v

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
        compute_function=compute_diff_vtheta_profile,
    )


def VerticalShearGradient(
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
    def compute_vertical_shear_gradient(helper):
        if _HAS_NUMBA:
            if shamrock.sys.world_rank() == 0:
                print("Using numba for custom getter in VerticalShearGradient")

        def internal(
            size: int, x: np.array, y: np.array, vx: np.array, vy: np.array, vz: np.array
        ) -> np.array:
            r = np.sqrt(x**2 + y**2)
            r_safe = r + 1e-9
            v_theta = (-y * vx + x * vy) / r_safe
            return v_theta

        if _HAS_NUMBA:
            internal = njit(internal)

        def custom_getter(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
                dic_out["vxyz"][:, 0],
                dic_out["vxyz"][:, 1],
                dic_out["vxyz"][:, 2],
            )

        arr_v_theta = helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=custom_getter,
        )

        extent = helper.get_extent()
        dy = (extent[3] - extent[2]) / helper.ny

        vert_shear_gradient = np.gradient(arr_v_theta, dy, axis=0)  # / dy

        return vert_shear_gradient

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
        compute_function=compute_vertical_shear_gradient,
    )


def gen_angular_momt_custom_getter(model, velocity_profile):
    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    if _HAS_NUMBA:
        if shamrock.sys.world_rank() == 0:
            print(
                "Using numba for velocity profile in SliceAngularMomentumTransportCoefficientPlot"
            )
        vel_profile_jit = njit(velocity_profile)
    else:
        vel_profile_jit = np.vectorize(velocity_profile)

    def internal(
        x: np.array,
        y: np.array,
        z: np.array,
        vx: np.array,
        vy: np.array,
        vz: np.array,
        hpart: np.array,
        cs: np.array,
    ) -> np.array:
        rho = pmass * (hfact / hpart) ** 3
        P = cs**2 * rho  # TODO: use true pressure

        r = np.sqrt(x**2 + y**2)
        r_safe = r + 1e-9
        v_r = (x * vx + y * vy) / r_safe
        v_theta = (-y * vx + x * vy) / r_safe

        delta_vtheta = v_theta - vel_profile_jit(r)
        alpha = rho * v_r * delta_vtheta / P

        return alpha

    if _HAS_NUMBA:
        if shamrock.sys.world_rank() == 0:
            print("Using numba for custom getter in SliceAngularMomentumTransportCoefficientPlot")
        internal = njit(internal)

    def custom_getter(size: int, dic_out: dict) -> np.array:
        return internal(
            dic_out["xyz"][:, 0],
            dic_out["xyz"][:, 1],
            dic_out["xyz"][:, 2],
            dic_out["vxyz"][:, 0],
            dic_out["vxyz"][:, 1],
            dic_out["vxyz"][:, 2],
            dic_out["hpart"],
            dic_out["soundspeed"],
        )

    return custom_getter


def SliceAngularMomentumTransportCoefficientPlot(
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
    velocity_profile=None,
):
    def compute_angular_momentum_transport_coefficient(helper):
        custom_getter = gen_angular_momt_custom_getter(model, velocity_profile)

        arr_v = helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=custom_getter,
        )

        return arr_v

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
        compute_function=compute_angular_momentum_transport_coefficient,
    )


def ColumnAverageAngularMomentumTransportCoefficientPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    min_normalization=1e-9,
    velocity_profile=None,
):
    def compute_angular_momentum_transport_coefficient(helper):
        custom_getter = gen_angular_momt_custom_getter(model, velocity_profile)

        arr_v = helper.column_average_render(
            "custom",
            "f64",
            min_normalization,
            custom_getter=custom_getter,
        )
        return arr_v

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
        compute_function=compute_angular_momentum_transport_coefficient,
    )
