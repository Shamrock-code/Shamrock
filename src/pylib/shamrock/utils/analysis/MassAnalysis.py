import json
import os

import numpy as np

import shamrock

try:
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class MassAnalysis:
    """
    Analysis utility to report the disc (gas + dust) mass history during the simulation.
    """

    def __init__(
        self, model, analysis_folder, analysis_prefix, time_unit="yr", mass_unit="sol_mass"
    ):
        self.model = model
        self.time_unit = time_unit
        self.mass_unit = mass_unit

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix)
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix)

        self.json_data_filename = self.analysis_prefix + ".json"
        self.plot_filename = self.plot_prefix

    def analysis_save(self, iplot):
        model = self.model

        solver_config = model.get_current_config().to_json()
        has_dust = not (solver_config["dust_config"]["mode"]["type"] == "none")

        model_units = model.get_current_config().get_units()

        barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

        mass_hist_new = {
            "time": self.model.get_time(),
            "disc_mass": disc_mass,
            "barycenter": barycenter,
            "unit_system": model_units.to_json(),
        }

        if has_dust:
            dust_mass = shamrock.model_sph.analysisDustMass(model=model).get_dust_mass()
            mass_hist_new["dust_mass"] = dust_mass

            drag_mode = solver_config["dust_config"]["drag_mode"]

            if "grains_sizes" in drag_mode:
                mass_hist_new["grains_sizes"] = drag_mode["grains_sizes"]

        if shamrock.sys.world_rank() == 0:
            try:
                with open(self.json_data_filename, "r") as fp:
                    mass_hist = json.load(fp)
            except (FileNotFoundError, json.JSONDecodeError):
                mass_hist = {"history": []}

            mass_hist["history"] = mass_hist["history"][:iplot] + [mass_hist_new]

            with open(self.json_data_filename, "w") as fp:
                print(f"Saving mass history to {self.json_data_filename}")
                json.dump(mass_hist, fp, indent=4)

    def load_analysis(self):
        with open(self.json_data_filename, "r") as fp:
            mass_hist = json.load(fp)
        return mass_hist

    def digest_perf_history(self, remove_null, smooth_window):
        mass_hist = self.load_analysis()

        has_dust = any("dust_mass" in h for h in mass_hist["history"])

        unit_system_json = next(
            (h["unit_system"] for h in mass_hist["history"] if "unit_system" in h), None
        )
        if unit_system_json is not None:
            model_units = shamrock.UnitSystem(**unit_system_json)
            time_conv = model_units.to(self.time_unit)
            mass_conv = model_units.to(self.mass_unit)
            length_conv = model_units.to("m")
        else:
            time_conv = 1.0
            mass_conv = 1.0
            length_conv = 1.0

        def rolling_smooth(arr, axis=0):
            # Average each sample with its neighbors within smooth_window
            # (a duration, in time_unit) of it in time.
            if smooth_window is None or arr.shape[axis] < 2:
                return arr
            half_window = smooth_window / 2.0
            smoothed = np.empty_like(arr, dtype=float)
            for i in range(t.shape[0]):
                mask = np.abs(t - t[i]) <= half_window
                smoothed[i] = np.mean(np.take(arr, np.nonzero(mask)[0], axis=axis), axis=axis)
            return smoothed

        def time_gradient(arr, axis=0):
            # np.gradient needs at least 2 samples along the gradient axis
            if t.shape[0] < 2:
                return np.array([])
            return rolling_smooth(np.gradient(arr, t, axis=axis), axis=axis)

        t = [h["time"] for h in mass_hist["history"]]
        disc_mass = [h["disc_mass"] for h in mass_hist["history"]]

        t = np.array(t) * time_conv
        disc_mass = np.array(disc_mass) * mass_conv

        result = {
            "t": t,
            "disc_mass": disc_mass,
            "d_disc_mass_dt": time_gradient(disc_mass),
            "delta_disc_mass": (disc_mass - disc_mass[0] if disc_mass.size > 0 else np.array([])),
            "delta_disc_mass_rel": (
                disc_mass / disc_mass[0] - 1 if disc_mass.size > 0 else np.array([])
            ),
        }

        if has_dust:
            dust_mass = [h["dust_mass"] for h in mass_hist["history"]]

            if remove_null:
                dust_mass = [[np.nan for v in dm] if np.max(dm) == 0 else dm for dm in dust_mass]

            dust_mass = np.array(dust_mass) * mass_conv
            dust_mass_all = np.sum(dust_mass, axis=-1)
            gas_mass = disc_mass - dust_mass_all

            result["dust_mass"] = dust_mass
            result["dust_mass_all"] = dust_mass_all
            result["gas_mass"] = gas_mass

            grains_sizes = next(
                h["grains_sizes"] for h in mass_hist["history"] if "grains_sizes" in h
            )
            result["grains_sizes"] = np.array(grains_sizes) * length_conv

            result["d_dust_mass_dt"] = time_gradient(dust_mass)
            result["d_dust_mass_all_dt"] = time_gradient(dust_mass_all)
            result["d_gas_mass_dt"] = time_gradient(gas_mass)

            result["delta_gas_mass"] = gas_mass - gas_mass[0] if gas_mass.size > 0 else gas_mass
            result["delta_gas_mass_rel"] = (
                gas_mass / gas_mass[0] - 1 if gas_mass.size > 0 else gas_mass
            )

            # Dust starts at zero before injection, so its delta is measured from the
            # first snapshot where the total dust mass is non null rather than t=0.
            dust_start_idx = self._first_nonnull_index(dust_mass_all)
            result["dust_start_idx"] = dust_start_idx

            if dust_start_idx is not None:
                result["delta_dust_mass_all"] = dust_mass_all - dust_mass_all[dust_start_idx]
                result["delta_dust_mass"] = dust_mass - dust_mass[dust_start_idx]
                result["delta_dust_mass_all_rel"] = (
                    dust_mass_all / dust_mass_all[dust_start_idx] - 1
                )
                result["delta_dust_mass_rel"] = dust_mass / dust_mass[dust_start_idx] - 1

        return result

    @staticmethod
    def _savefig(fig, filename_png):
        """Save a figure as png, and also as pdf at the same path."""
        fig.savefig(filename_png)
        fig.savefig(filename_png.replace(".png", ".pdf"))

    @staticmethod
    def _first_nonnull_index(arr):
        """Index of the first entry that is neither NaN nor zero, or None if there is none."""
        valid = ~np.isnan(arr) & (arr != 0)
        if not valid.any():
            return None
        return int(np.argmax(valid))

    @staticmethod
    def _symlog_linthresh(*arrays):
        """Pick a symlog linear threshold from the smallest data scale present."""
        abs_vals = np.concatenate([np.abs(a).ravel() for a in arrays if a.size > 0])
        abs_vals = abs_vals[np.isfinite(abs_vals) & (abs_vals > 0)]
        if abs_vals.size == 0:
            return 1e-10
        return np.nanmax(abs_vals) * 1e-6

    @staticmethod
    def _clamp_log_ylim(ax, *arrays, decades=10, margin_frac=0.1):
        """Clamp a log-scale axis to at most `decades` below the max, so a handful of
        near-zero samples don't stretch the y-axis over many empty decades. A margin
        (in log space) is added on both sides to avoid clamping the plotted curves."""
        vals = np.concatenate([np.asarray(a).ravel() for a in arrays if np.size(a) > 0])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return
        ymin, ymax = vals.min(), vals.max()
        if ymin >= ymax * 10**-decades:
            return
        margin = margin_frac * decades
        ax.set_ylim(ymax * 10 ** (-decades - margin), ymax * 10**margin)

    def _plot_mass_delta(
        self,
        mass_hist,
        t,
        key_suffix,
        ylabel,
        filename_suffix,
        ndust,
        dust_colors,
        dust_cmap,
        dust_norm,
        figsize,
        dpi,
        close_plots,
    ):
        has_dust = f"delta_gas_mass{key_suffix}" in mass_hist
        has_dust_delta = mass_hist.get("dust_start_idx") is not None

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.gca()
        ax.plot(t, mass_hist[f"delta_disc_mass{key_suffix}"], color="0.0", label="$M$")

        linthresh_arrays = [mass_hist[f"delta_disc_mass{key_suffix}"]]

        if has_dust:
            ax.plot(
                t,
                mass_hist[f"delta_gas_mass{key_suffix}"],
                color="cornflowerblue",
                label=r"$M_{\rm gas}$",
            )
            linthresh_arrays.append(mass_hist[f"delta_gas_mass{key_suffix}"])

        if has_dust_delta:
            ax.plot(
                t,
                mass_hist[f"delta_dust_mass_all{key_suffix}"],
                color="0.5",
                label=r"$M_{\rm dust}$",
            )
            for i in range(ndust):
                ax.plot(t, mass_hist[f"delta_dust_mass{key_suffix}"][:, i], color=dust_colors[i])
            linthresh_arrays += [
                mass_hist[f"delta_dust_mass_all{key_suffix}"],
                mass_hist[f"delta_dust_mass{key_suffix}"],
            ]

        ax.set_xlabel(f"t [{self.time_unit}]")
        ax.set_ylabel(ylabel)
        ax.set_yscale("symlog", linthresh=self._symlog_linthresh(*linthresh_arrays))

        handles, labels = ax.get_legend_handles_labels()
        if has_dust_delta:
            shamrock.matplotlib.add_cmap_legend_entry(
                ax,
                dust_cmap,
                label=r"$M_{\rm dust}(s_{\rm grain})$",
                extra_handles=handles,
                extra_labels=labels,
                loc="best",
            )

            dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
            dust_sm.set_array([])
            cbar = fig.colorbar(dust_sm, ax=ax)
            cbar.set_label("grain size [m]")
        else:
            ax.legend(loc="best")

        self._savefig(fig, self.plot_filename + filename_suffix)
        if close_plots:
            plt.close(fig)

    def plot_history(
        self, close_plots=True, figsize=(8, 5), dpi=200, remove_null=True, smooth_window=None
    ):
        if not _HAS_MATPLOTLIB:
            print("Warning: matplotlib is not installed, plot_perf_history is a no-op")
            return

        if shamrock.sys.world_rank() == 0:
            mass_hist = self.digest_perf_history(
                remove_null=remove_null, smooth_window=smooth_window
            )

            print(f"Plotting mass history from {self.json_data_filename}")

            t = mass_hist["t"]

            mass_unit_text = self.mass_unit
            if mass_unit_text == "sol_mass":
                mass_unit_text = "sol mass"

            fig = plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, mass_hist["disc_mass"])
            plt.xlabel(f"t [{self.time_unit}]")
            plt.ylabel(f"total mass [{mass_unit_text}]")
            self._savefig(fig, self.plot_filename + "_total_mass.png")
            if close_plots:
                plt.close(fig)

            if mass_hist["d_disc_mass_dt"].size > 0:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(t, mass_hist["d_disc_mass_dt"])
                plt.xlabel(f"t [{self.time_unit}]")
                plt.ylabel(rf"$dM/dt$ [{mass_unit_text} / {self.time_unit}]")
                plt.yscale("symlog", linthresh=self._symlog_linthresh(mass_hist["d_disc_mass_dt"]))
                self._savefig(fig, self.plot_filename + "_total_mass_dot.png")
                if close_plots:
                    plt.close(fig)

            ndust = dust_colors = dust_cmap = dust_norm = None

            if "dust_mass" in mass_hist:
                ndust = mass_hist["dust_mass"].shape[-1]

                grains_sizes = mass_hist["grains_sizes"]

                dust_cmap = plt.colormaps["plasma"]
                dust_vmin = 10 ** np.floor(np.log10(grains_sizes.min()))
                dust_vmax = 10 ** np.ceil(np.log10(grains_sizes.max()))
                dust_norm = mcolors.LogNorm(vmin=dust_vmin, vmax=dust_vmax)
                dust_colors = dust_cmap(dust_norm(grains_sizes))

                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = fig.gca()
                ax.plot(t, mass_hist["disc_mass"], color="0.0", label="$M$")
                ax.plot(t, mass_hist["gas_mass"], color="cornflowerblue", label=r"$M_{\rm gas}$")
                ax.plot(t, mass_hist["dust_mass_all"], color="0.5", label=r"$M_{\rm dust}$")
                for i in range(ndust):
                    ax.plot(t, mass_hist["dust_mass"][:, i], color=dust_colors[i])

                ax.set_xlabel(f"t [{self.time_unit}]")
                ax.set_ylabel(f"mass [{mass_unit_text}]")
                ax.set_yscale("log")
                self._clamp_log_ylim(
                    ax,
                    mass_hist["disc_mass"],
                    mass_hist["gas_mass"],
                    mass_hist["dust_mass_all"],
                    mass_hist["dust_mass"],
                )
                handles, labels = ax.get_legend_handles_labels()
                shamrock.matplotlib.add_cmap_legend_entry(
                    ax,
                    dust_cmap,
                    label=r"$M_{\rm dust}(s_{\rm grain})$",
                    extra_handles=handles,
                    extra_labels=labels,
                    loc="best",
                )

                dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
                dust_sm.set_array([])
                cbar = fig.colorbar(dust_sm, ax=ax)
                cbar.set_label("grain size [m]")

                self._savefig(fig, self.plot_filename + "_masses.png")
                if close_plots:
                    plt.close(fig)

                if mass_hist["d_dust_mass_dt"].size > 0:
                    fig = plt.figure(figsize=figsize, dpi=dpi)
                    ax = fig.gca()
                    ax.plot(t, mass_hist["d_disc_mass_dt"], color="0.0", label=r"$\dot{M}$")
                    ax.plot(
                        t,
                        mass_hist["d_gas_mass_dt"],
                        "-",
                        color="cornflowerblue",
                        label=r"$\dot{M}_{\rm gas}$",
                    )
                    ax.plot(
                        t,
                        mass_hist["d_dust_mass_all_dt"],
                        color="0.5",
                        label=r"$\dot{M}_{\rm dust}$",
                    )
                    for i in range(ndust):
                        ax.plot(t, mass_hist["d_dust_mass_dt"][:, i], color=dust_colors[i])

                    ax.set_xlabel(f"t [{self.time_unit}]")
                    ax.set_ylabel(rf"$dM/dt$ [{mass_unit_text} / {self.time_unit}]")
                    ax.set_yscale(
                        "symlog",
                        linthresh=self._symlog_linthresh(
                            mass_hist["d_disc_mass_dt"],
                            mass_hist["d_gas_mass_dt"],
                            mass_hist["d_dust_mass_all_dt"],
                            mass_hist["d_dust_mass_dt"],
                        ),
                    )
                    handles, labels = ax.get_legend_handles_labels()
                    shamrock.matplotlib.add_cmap_legend_entry(
                        ax,
                        dust_cmap,
                        label=r"$\dot{M}_{\rm dust}(s_{\rm grain})$",
                        extra_handles=handles,
                        extra_labels=labels,
                        loc="best",
                    )

                    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
                    dust_sm.set_array([])
                    cbar = fig.colorbar(dust_sm, ax=ax)
                    cbar.set_label("grain size [m]")

                    self._savefig(fig, self.plot_filename + "_masses_dot.png")
                    if close_plots:
                        plt.close(fig)

            # Mass conservation since the simulation start, in both absolute and relative
            # terms. Dust (and everything derived from it) is instead measured relative to
            # the first snapshot where the total dust mass is non null, since dust starts
            # at zero.
            common_kwargs = dict(
                mass_hist=mass_hist,
                t=t,
                ndust=ndust,
                dust_colors=dust_colors,
                dust_cmap=dust_cmap,
                dust_norm=dust_norm,
                figsize=figsize,
                dpi=dpi,
                close_plots=close_plots,
            )

            self._plot_mass_delta(
                key_suffix="",
                ylabel=rf"$\delta M$ [{mass_unit_text}]",
                filename_suffix="_mass_delta.png",
                **common_kwargs,
            )
            self._plot_mass_delta(
                key_suffix="_rel",
                ylabel=r"$\delta M / M_0$",
                filename_suffix="_mass_delta_rel.png",
                **common_kwargs,
            )
