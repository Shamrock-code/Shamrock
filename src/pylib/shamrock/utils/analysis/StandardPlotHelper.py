import glob
import json
import os

import numpy as np

import shamrock.sys

from .UnitHelper import plot_codeu_to_unit

try:
    import matplotlib
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def analysis_save(iplot, data, metadata, npy_data_filename, json_data_filename):
    """
    Save the analysis data to the json and npy files
    """
    if shamrock.sys.world_rank() == 0:
        print(f"Saving data to {npy_data_filename.format(iplot)}")
        np.save(npy_data_filename.format(iplot), data)

        with open(json_data_filename.format(iplot), "w") as fp:
            print(f"Saving metadata to {json_data_filename.format(iplot)}")
            json.dump(metadata, fp)


def load_analysis(iplot, json_data_filename, npy_data_filename):
    """
    Load the analysis data from the json and npy files
    """
    with open(json_data_filename.format(iplot), "r") as fp:
        metadata = json.load(fp)
    return np.load(npy_data_filename.format(iplot)), metadata


def get_list_analysis_id(glob_str_data):
    """
    Get the list of analysis ids from the glob string
    """
    list_files = glob.glob(glob_str_data)
    list_files.sort()
    list_analysis_id = []
    for f in list_files:
        list_analysis_id.append(int(f.split("_")[-1].split(".")[0]))
    return list_analysis_id


def field_normalize(field, normalization, min_normalization=1e-9):
    """
    Normalize the field by the normalization and set to nan below min_normalization
    """
    ret = field / normalization

    # set to nan below min_normalization
    ret[normalization < min_normalization] = np.nan

    return ret


def figure_init(aspect, holywood_mode=False, dpi=200, base_size=6, nx=None, ny=None):
    """
    Initialize the figure
    """
    figsize = (aspect * base_size, 1.0 * base_size)

    if not holywood_mode:
        fx, fy = figsize
        figsize = (fx + 1, fy)

    # Reset the figure using the same memory as the last one
    plt.figure(figsize=figsize, num=1, clear=True, dpi=dpi)

    if holywood_mode:
        if nx is None or ny is None:
            raise ValueError("nx and ny must be provided in holywood mode")

        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(nx / dpi, ny / dpi)
        plt.axis("off")


def figure_add_colorbar(imshow_result, label, holywood_mode=False):
    """
    Add the colorbar to the figure
    """
    if holywood_mode:
        axins = plt.gca().inset_axes([0.73, 0.1, 0.25, 0.025])
        cbar = plt.colorbar(imshow_result, cax=axins, orientation="horizontal", extend="both")
        cbar.set_label(label, color="white")

        # Set colorbar elements to white
        cbar.outline.set_edgecolor("white")
        # cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.get_yticklabels(), color="white")
        plt.setp(cbar.ax.get_xticklabels(), color="white")
        cbar.ax.tick_params(color="white", labelcolor="white", length=6, width=1)

    else:
        cbar = plt.colorbar(imshow_result, extend="both")
        cbar.set_label(label)


def figure_render_sinks(sink_pos_screen, ax, scale_factor, color, linewidth, fill):
    """
    Render the sinks on the figure
    """
    output_list = []
    for x, y, s in sink_pos_screen:
        output_list.append(
            plt.Circle(
                (x, y),
                s["accretion_radius"] * scale_factor,
                linewidth=linewidth,
                color=color,
                fill=fill,
            )
        )
    for circle in output_list:
        ax.add_artist(circle)


def figure_add_time_info(text, holywood_mode=False):
    """
    Add the time info to the figure
    """
    if holywood_mode:
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=2)
        plt.gca().add_artist(anchored_text)
    else:
        plt.title(text)


def init_analysis_plot_paths(obj, analysis_folder, analysis_prefix):
    plots_dir = os.path.join(analysis_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    obj.analysis_prefix = os.path.join(plots_dir, analysis_prefix) + "_"
    obj.plot_prefix = os.path.join(plots_dir, "plot_" + analysis_prefix) + "_"

    obj.npy_data_filename = obj.analysis_prefix + "{:07}.npy"
    obj.json_data_filename = obj.analysis_prefix + "{:07}.json"
    obj.plot_filename = obj.plot_prefix + "{:07}.png"
    obj.glob_str_plot = obj.plot_prefix + "*.png"
    obj.glob_str_data = obj.analysis_prefix + "*.json"


class StandardPlotHelper:
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
        compute_function=None,
    ):
        self.model = model
        self.ext_r = ext_r
        self.nx = nx
        self.ny = ny
        self.ex = ex
        self.ey = ey
        self.center = center
        self.aspect = float(self.nx) / float(self.ny)
        self.compute_function = compute_function
        init_analysis_plot_paths(self, analysis_folder, analysis_prefix)

    def get_dx_dy(self):
        ext_x = 2 * self.ext_r * self.aspect
        ext_y = 2 * self.ext_r

        dx = (self.ex[0] * ext_x, self.ex[1] * ext_x, self.ex[2] * ext_x)
        dy = (self.ey[0] * ext_y, self.ey[1] * ext_y, self.ey[2] * ext_y)

        return dx, dy

    def column_integ_render(self, field_name, field_type, custom_getter=None):
        dx, dy = self.get_dx_dy()
        arr_field = self.model.render_cartesian_column_integ(
            field_name,
            field_type,
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
            custom_getter=custom_getter,
        )

        return arr_field

    def column_average_render(
        self, field_name, field_type, min_normalization=1e-9, custom_getter=None
    ):
        dx, dy = self.get_dx_dy()
        arr_field = self.model.render_cartesian_column_integ(
            field_name,
            field_type,
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
            custom_getter=custom_getter,
        )

        normalisation = self.model.render_cartesian_column_integ(
            "unity",
            "f64",
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
        )

        return field_normalize(arr_field, normalisation, min_normalization)

    def slice_render(
        self,
        field_name,
        field_type,
        do_normalization=True,
        min_normalization=1e-9,
        field_transform=None,
        custom_getter=None,
    ):
        dx, dy = self.get_dx_dy()
        arr_field_data = self.model.render_cartesian_slice(
            field_name,
            field_type,
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
            custom_getter=custom_getter,
        )

        if field_transform is not None:
            arr_field_data = field_transform(arr_field_data)

        if not do_normalization:
            return arr_field_data

        arr_field_normalization = self.model.render_cartesian_slice(
            "unity",
            "f64",
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
        )

        return field_normalize(arr_field_data, arr_field_normalization, min_normalization)

    def get_extent(self):
        x_e_x = (
            self.ex[0] * self.center[0] + self.ex[1] * self.center[1] + self.ex[2] * self.center[2]
        )
        y_e_y = (
            self.ey[0] * self.center[0] + self.ey[1] * self.center[1] + self.ey[2] * self.center[2]
        )
        return [
            -self.ext_r * self.aspect + x_e_x,
            self.ext_r * self.aspect + x_e_x,
            -self.ext_r + y_e_y,
            self.ext_r + y_e_y,
        ]

    def compute(self):
        if self.compute_function is None:
            raise ValueError("compute_function is not set")
        return self.compute_function(self)

    def analysis_save(self, iplot, data=None):
        metadata = {
            "extent": self.get_extent(),
            "time": self.model.get_time(),
            "sinks": self.model.get_sinks(),
        }
        if data is None:
            data = self.compute()
        analysis_save(iplot, data, metadata, self.npy_data_filename, self.json_data_filename)

    def load_analysis(self, iplot):
        return load_analysis(iplot, self.json_data_filename, self.npy_data_filename)

    def get_list_analysis_id(self):
        return get_list_analysis_id(self.glob_str_data)

    def metadata_to_screen_sink_pos(self, metadata):
        output_list = []
        for s in metadata["sinks"]:
            # print(s)
            x, y, z = s["pos"]

            x_e_x = self.ex[0] * x + self.ex[1] * y + self.ex[2] * z
            y_e_y = self.ey[0] * x + self.ey[1] * y + self.ey[2] * z

            output_list.append((x_e_x, y_e_y, s))
        return output_list

    def figure_init(self, holywood_mode=False, dpi=200):
        figure_init(self.aspect, holywood_mode, dpi, base_size=6, nx=self.nx, ny=self.ny)

    def figure_render_sinks(self, metadata, ax, scale_factor, color, linewidth, fill):
        sink_list_plot = self.metadata_to_screen_sink_pos(metadata)
        figure_render_sinks(sink_list_plot, ax, scale_factor, color, linewidth, fill)

    def figure_add_time_info(self, text, holywood_mode=False):
        figure_add_time_info(text, holywood_mode)

    def figure_add_colorbar(self, imshow_result, label, holywood_mode=False):
        figure_add_colorbar(imshow_result, label, holywood_mode)

    def make_plot(
        self,
        iplot,
        x_unit=None,
        y_unit=None,
        time_unit=None,
        field_unit=None,
        x_label=None,
        y_label=None,
        field_label=None,
        holywood_mode=False,
        cmap="magma",
        add_sinks=True,
        sink_scale_factor=1,
        sink_color="green",
        sink_linewidth=1,
        sink_fill=False,
        save_plot=True,
        **kwargs,
    ):
        if shamrock.sys.world_rank() == 0:
            field_render, metadata = self.load_analysis(iplot)

            dist_label_x, dist_conv_x = plot_codeu_to_unit(self.model.get_units(), x_unit)
            dist_label_y, dist_conv_y = plot_codeu_to_unit(self.model.get_units(), y_unit)

            metadata["extent"][0] *= dist_conv_x
            metadata["extent"][1] *= dist_conv_x
            metadata["extent"][2] *= dist_conv_y
            metadata["extent"][3] *= dist_conv_y

            time_label, time_conv = plot_codeu_to_unit(self.model.get_units(), time_unit)
            metadata["time"] *= time_conv

            field_unit_label, field_conv = plot_codeu_to_unit(self.model.get_units(), field_unit)
            field_render *= field_conv

            self.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps[cmap].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            res = plt.imshow(
                field_render, cmap=my_cmap, origin="lower", extent=metadata["extent"], **kwargs
            )

            ax = plt.gca()

            if add_sinks:
                self.figure_render_sinks(
                    metadata, ax, sink_scale_factor, sink_color, sink_linewidth, sink_fill
                )

            plt.xlabel(f"{x_label} {dist_label_x}")
            plt.ylabel(f"{y_label} {dist_label_y}")

            text = f"t = {metadata['time']:0.3f} {time_label}"
            self.figure_add_time_info(text, holywood_mode)

            cmap_label = f"{field_label} {field_unit_label}"
            self.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.plot_filename.format(iplot)}")
            plt.savefig(self.plot_filename.format(iplot))
            plt.close()

    def render_all(self, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.make_plot(iplot, **kwargs)

    def render_gif(self, gif_filename, save_animation=False, fps=15, bitrate=1800):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(self.glob_str_plot, render_gif=True)
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(
                    fps=fps, metadata=dict(artist="Me"), bitrate=bitrate
                )
                ani.save(self.analysis_prefix + gif_filename, writer=writer)
            return ani
        return None
