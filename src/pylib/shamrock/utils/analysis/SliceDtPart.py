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


class SliceDtPart:
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

    def compute_dt_part(self):
        if not self.model.get_current_config().should_save_dt_to_fields():
            raise ValueError("dt_part is not saved to fields")

        arr_dt_part = self.helper.slice_render(
            "dt_part",
            "f64",
            self.do_normalization,
            self.min_normalization,
        )

        return arr_dt_part

    def analysis_save(self, iplot):
        arr_dt_part = self.compute_dt_part()
        self.helper.analysis_save(iplot, arr_dt_part)

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
        contour_list=None,
        **kwargs,
    ):
        if shamrock.sys.world_rank() == 0:
            arr_dt_part, metadata = self.load_analysis(iplot)

            dist_label, dist_conv = plot_codeu_to_unit(self.model.get_units(), dist_unit)
            metadata["extent"] = [metadata["extent"][i] * dist_conv for i in range(4)]

            time_label, time_conv = plot_codeu_to_unit(self.model.get_units(), time_unit)
            metadata["time"] *= time_conv

            dt_part_label, dt_part_conv = plot_codeu_to_unit(self.model.get_units(), time_unit)
            arr_dt_part *= dt_part_conv

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["magma"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            # Create coordinate arrays matching the extent for contour alignment
            ny, nx = arr_dt_part.shape
            x = np.linspace(metadata["extent"][0], metadata["extent"][1], nx)
            y = np.linspace(metadata["extent"][2], metadata["extent"][3], ny)
            X, Y = np.meshgrid(x, y)

            # Draw contours and add labels
            if contour_list is not None:
                contour_set = plt.contour(
                    X, Y, arr_dt_part, levels=contour_list, colors="white", linewidths=0.5
                )
                plt.clabel(contour_set, inline=True, fontsize=8, fmt="%g")

            res = plt.imshow(
                arr_dt_part,
                cmap=my_cmap,
                origin="lower",
                extent=metadata["extent"],
                **kwargs,
            )

            ax = plt.gca()

            self.helper.figure_render_sinks(metadata, ax)

            plt.xlabel(f"x {dist_label}")
            plt.ylabel(f"y {dist_label}")

            text = f"t = {metadata['time']:0.3f} {time_label}"
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = f"$\\Delta t$ {dt_part_label}"
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
        gif_filename="dt_part_slice.gif",
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
