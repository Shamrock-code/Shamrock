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

from .StandardPlotHelper import StandardPlotHelper
from .UnitHelper import plot_codeu_to_unit


class SliceDensityPlot:
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

    def compute_rho_xy(self):
        arr_rho_xy = self.helper.slice_render(
            "rho", "f64", self.do_normalization, self.min_normalization
        )

        return arr_rho_xy

    def analysis_save(self, iplot):
        arr_rho_xy = self.compute_rho_xy()
        self.helper.analysis_save(iplot, arr_rho_xy)

    def load_analysis(self, iplot):
        return self.helper.load_analysis(iplot)

    def get_list_analysis_id(self):
        return self.helper.get_list_analysis_id()

    def plot_rho_xy(
        self,
        iplot,
        holywood_mode=False,
        dist_unit="au",
        time_unit="year",
        density_unit="kg.m^-3",
        **kwargs,
    ):
        if shamrock.sys.world_rank() == 0:
            arr_rho_xy, metadata = self.load_analysis(iplot)

            dist_label, dist_conv = plot_codeu_to_unit(self.model.get_units(), dist_unit)
            metadata["extent"] = [metadata["extent"][i] * dist_conv for i in range(4)]

            time_label, time_conv = plot_codeu_to_unit(self.model.get_units(), time_unit)
            metadata["time"] *= time_conv

            density_label, density_conv = plot_codeu_to_unit(self.model.get_units(), density_unit)
            arr_rho_xy *= density_conv

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["magma"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            res = plt.imshow(
                arr_rho_xy, cmap=my_cmap, origin="lower", extent=metadata["extent"], **kwargs
            )

            ax = plt.gca()

            self.helper.figure_render_sinks(metadata, ax)

            plt.xlabel(f"x {dist_label}")
            plt.ylabel(f"y {dist_label}")

            text = f"t = {metadata['time']:0.3f} {time_label}"
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = f"$\\rho$ {density_label}"
            self.helper.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.helper.plot_filename.format(iplot)}")
            plt.savefig(self.helper.plot_filename.format(iplot))
            plt.close()

    def render_all(self, holywood_mode=False, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.plot_rho_xy(iplot, holywood_mode, **kwargs)

    def render_gif(
        self,
        save_animation=False,
        fps=15,
        bitrate=1800,
        gif_filename="rho_slice.gif",
    ):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(
                self.helper.glob_str_plot, render_gif=True
            )
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(
                    fps=fps, metadata=dict(artist="Shamrock"), bitrate=bitrate
                )
                ani.save(self.helper.analysis_prefix + gif_filename, writer=writer)
            return ani
        return None
