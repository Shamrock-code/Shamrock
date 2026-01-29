import json
import os

import numpy as np

import shamrock.sys

try:
    import matplotlib
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from .StandardPlotHelper import StandardPlotHelper
from .UnitHelper import plot_codeu_to_unit


class VerticalShearGradient:
    def __init__(
        self,
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
        self.model = model
        self.helper = StandardPlotHelper(
            model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix
        )
        self.do_normalization = do_normalization
        self.min_normalization = min_normalization

    def compute_vertical_shear_gradient(self):
        if _HAS_NUMBA:
            if shamrock.sys.world_rank() == 0:
                print("Using numba for custom getter in VerticalShearGradient")

        def internal(
            size: int, x: np.array, y: np.array, vx: np.array, vy: np.array, vz: np.array
        ) -> np.array:
            v_theta = np.zeros(size)
            for i in range(size):
                e_theta = np.array([-y[i], x[i], 0])
                e_theta /= np.linalg.norm(e_theta) + 1e-9  # Avoid division by zero
                v_theta[i] = np.dot(e_theta, np.array([vx[i], vy[i], vz[i]]))
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

        arr_v_theta = self.helper.slice_render(
            "custom",
            "f64",
            self.do_normalization,
            self.min_normalization,
            custom_getter=custom_getter,
        )

        extent = self.helper.get_extent()
        dy = (extent[3] - extent[2]) / self.helper.ny

        vert_shear_gradient = np.gradient(arr_v_theta, dy, axis=0)  # / dy

        return vert_shear_gradient

    def analysis_save(self, iplot):
        vert_shear_gradient = self.compute_vertical_shear_gradient()
        self.helper.analysis_save(iplot, vert_shear_gradient)

    def load_analysis(self, iplot):
        return self.helper.load_analysis(iplot)

    def get_list_analysis_id(self):
        return self.helper.get_list_analysis_id()

    def plot(
        self,
        iplot,
        holywood_mode=False,
        dist_unit="au",
        time_unit="year",
        sink_scale_factor=1,
        sink_color="green",
        sink_linewidth=1,
        sink_fill=False,
        **kwargs,
    ):
        if shamrock.sys.world_rank() == 0:
            vert_shear_gradient, metadata = self.load_analysis(iplot)

            dist_label, dist_conv = plot_codeu_to_unit(self.model.get_units(), dist_unit)
            metadata["extent"] = [metadata["extent"][i] * dist_conv for i in range(4)]

            time_label, time_conv = plot_codeu_to_unit(self.model.get_units(), time_unit)
            metadata["time"] *= time_conv

            vert_shear_gradient_label, vert_shear_gradient_conv = plot_codeu_to_unit(
                self.model.get_units(), "yr^-1"
            )
            vert_shear_gradient *= vert_shear_gradient_conv

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["seismic"].copy()  # copy the default cmap
            my_cmap.set_bad(color="white")

            res = plt.imshow(
                vert_shear_gradient,
                cmap=my_cmap,
                origin="lower",
                extent=metadata["extent"],
                **kwargs,
            )

            ax = plt.gca()

            self.helper.figure_render_sinks(
                metadata, ax, sink_scale_factor, sink_color, sink_linewidth, sink_fill
            )

            plt.xlabel(f"x {dist_label}")
            plt.ylabel(f"y {dist_label}")

            text = f"t = {metadata['time']:0.3f} {time_label}"
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = f"${{\\partial R \\Omega}}/{{\\partial z}}$ {vert_shear_gradient_label}"
            self.helper.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.helper.plot_filename.format(iplot)}")
            plt.savefig(self.helper.plot_filename.format(iplot))
            plt.close()

    def render_all(self, holywood_mode=False, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.plot(iplot, holywood_mode, **kwargs)

    def render_gif(
        self,
        save_animation=False,
        fps=15,
        bitrate=1800,
        gif_filename="vertical_shear_gradient_slice.gif",
    ):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(
                self.helper.glob_str_plot, render_gif=True
            )
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(
                    fps=fps, metadata=dict(artist="Me"), bitrate=bitrate
                )
                ani.save(
                    self.helper.analysis_prefix + gif_filename,
                    writer=writer,
                )
            return ani
        return None
