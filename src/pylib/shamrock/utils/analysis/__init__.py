import json
import os

import numpy as np

import shamrock.sys

try:
    import matplotlib
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    print("Warning: matplotlib is not installed, some Shamrock functions will not be available")


class perf_history:
    def __init__(self, model, analysis_folder, analysis_prefix):
        self.model = model

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix)
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix)

        self.json_data_filename = self.analysis_prefix + ".json"
        self.plot_filename = self.plot_prefix

    def analysis_save(self, iplot):
        sim_time_delta = self.model.solver_logs_cumulated_step_time()
        scount = self.model.solver_logs_step_count()
        part_count = self.model.get_total_part_count()

        self.model.solver_logs_reset_cumulated_step_time()
        self.model.solver_logs_reset_step_count()

        if shamrock.sys.world_rank() == 0:
            perf_hist_new = {
                "time": self.model.get_time(),
                "sim_time_delta": sim_time_delta,
                "world_size": shamrock.sys.world_size(),
                "sim_step_count_delta": scount,
                "part_count": part_count,
            }

            try:
                with open(self.json_data_filename, "r") as fp:
                    perf_hist = json.load(fp)
            except (FileNotFoundError, json.JSONDecodeError):
                perf_hist = {"history": []}

            perf_hist["history"] = perf_hist["history"][:iplot] + [perf_hist_new]

            if scount == 0:
                print("Warning: step count is 0, skipping save of perf history")
                return

            with open(self.json_data_filename, "w") as fp:
                print(f"Saving perf history to {self.json_data_filename}")
                json.dump(perf_hist, fp, indent=4)

    def load_analysis(self):
        with open(self.json_data_filename, "r") as fp:
            perf_hist = json.load(fp)
        return perf_hist

    def digest_perf_history(self):
        perf_hist = self.load_analysis()

        t = [h["time"] for h in perf_hist["history"]]
        sim_time_delta = [h["sim_time_delta"] for h in perf_hist["history"]]
        world_size = [h["world_size"] for h in perf_hist["history"]]
        sim_step_count_delta = [h["sim_step_count_delta"] for h in perf_hist["history"]]
        part_count = [h["part_count"] for h in perf_hist["history"]]

        t = np.array(t)
        dt_code = np.diff(t)

        sim_time_delta = np.array(sim_time_delta)
        world_size = np.array(world_size)
        sim_time_delta_all_proc = sim_time_delta * world_size
        sim_step_count_delta = np.array(sim_step_count_delta)
        part_count = np.array(part_count)

        # cumulative sim_time & step_count
        cum_sim_time_delta = np.cumsum(sim_time_delta)
        cum_sim_time_delta_all_proc = np.cumsum(sim_time_delta_all_proc)
        cum_sim_step_count_delta = np.cumsum(sim_step_count_delta)

        tsim_per_hour = dt_code / (sim_time_delta[1:] / 3600)

        time_per_step = []

        for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
            if sc > 0:
                time_per_step.append(td / sc)
            else:
                # NAN here because the step count is 0
                time_per_step.append(np.nan)

        rate = []

        for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
            if sc > 0:
                rate.append(pc / (td / sc))
            else:
                # NAN here because the step count is 0
                rate.append(np.nan)

        return {
            "t": t,
            "dt_code": dt_code,
            "part_count": part_count,
            "world_size": world_size,
            "sim_time_delta": sim_time_delta,
            "sim_step_count_delta": sim_step_count_delta,
            "cum_sim_time_delta": cum_sim_time_delta,
            "cum_sim_time_delta_all_proc": cum_sim_time_delta_all_proc,
            "cum_sim_step_count_delta": cum_sim_step_count_delta,
            "time_per_step": time_per_step,
            "rate": rate,
            "tsim_per_hour": tsim_per_hour,
        }

    def plot_perf_history(self, close_plots=True, show_plots=False, figsize=(8, 5), dpi=200):
        if not _HAS_MATPLOTLIB:
            print("Warning: matplotlib is not installed, plot_perf_history is a no-op")
            return

        if shamrock.sys.world_rank() == 0:
            perf_hist = self.digest_perf_history()

            print(f"Plotting perf history from {self.json_data_filename}")

            t = perf_hist["t"]

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["cum_sim_time_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("t [s] (real time)")
            plt.savefig(self.plot_filename + "_sim_time.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(
                t,
                perf_hist["cum_sim_time_delta_all_proc"] / 3600.0,
                "+-",
                label="Used compute time",
            )
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$\sum_{processes} t$ [h] (real time)")

            ax1 = plt.gca()

            # Right y-axis
            ax2 = ax1.twinx()
            ax2.plot(t, perf_hist["world_size"], "+-", color="tab:orange", label="World size")
            ax2.set_ylabel("World size")

            # Optional: combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            plt.savefig(self.plot_filename + "_sim_time_all_proc.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["cum_sim_step_count_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$N_\\mathrm{step}$")
            plt.savefig(self.plot_filename + "_step_count.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["sim_time_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$d t_\\mathrm{real} / d i_\\mathrm{analysis}$ [s]")
            plt.savefig(self.plot_filename + "_sim_time_delta.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["sim_step_count_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$d N_\\mathrm{step} / d i_\\mathrm{analysis}$")
            plt.savefig(self.plot_filename + "_step_count_delta.png")
            if close_plots:
                plt.close()

            # tsim per hour
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t[1:], perf_hist["tsim_per_hour"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$d t_\\mathrm{sim} / d t_\\mathrm{realtime}$ [code unit (time) / hour]")
            plt.savefig(self.plot_filename + "_tsim_per_hour.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["time_per_step"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("time per step [s]")
            plt.savefig(self.plot_filename + "_time_per_step.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["rate"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("Particles / second")
            plt.yscale("log")
            plt.savefig(self.plot_filename + "_rate.png")
            if close_plots:
                plt.close()

            if show_plots:
                plt.show()
