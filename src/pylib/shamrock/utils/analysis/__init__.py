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

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix) + "_"
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix) + "_"

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
                "sim_step_count_delta": scount,
                "part_count": part_count,
            }

            try:
                with open(self.json_data_filename, "r") as fp:
                    perf_hist = json.load(fp)
            except (FileNotFoundError, json.JSONDecodeError):
                perf_hist = {"history": []}

            perf_hist["history"] = perf_hist["history"][:iplot] + [perf_hist_new]

            with open(self.json_data_filename, "w") as fp:
                print(f"Saving perf history to {self.json_data_filename}")
                json.dump(perf_hist, fp, indent=4)

    def load_analysis(self):
        with open(self.json_data_filename, "r") as fp:
            perf_hist = json.load(fp)
        return perf_hist

    def plot_perf_history(self, close_plots=True, show_plots=False):
        if shamrock.sys.world_rank() == 0:
            perf_hist = self.load_analysis()

            print(f"Plotting perf history from {self.json_data_filename}")

            t = [h["time"] for h in perf_hist["history"]]
            sim_time_delta = [h["sim_time_delta"] for h in perf_hist["history"]]
            sim_step_count_delta = [h["sim_step_count_delta"] for h in perf_hist["history"]]
            part_count = [h["part_count"] for h in perf_hist["history"]]

            t = np.array(t)
            sim_time_delta = np.array(sim_time_delta)
            sim_step_count_delta = np.array(sim_step_count_delta)
            part_count = np.array(part_count)

            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(t, sim_time_delta)
            plt.xlabel("t")
            plt.ylabel("sim_time_delta")
            plt.savefig(self.plot_filename + "_sim_time_delta.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(t, sim_step_count_delta)
            plt.xlabel("t")
            plt.ylabel("sim_step_count_delta")
            plt.savefig(self.plot_filename + "_step_count_delta.png")
            if close_plots:
                plt.close()

            time_per_step = []

            for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
                if sc > 0:
                    time_per_step.append(td / sc)
                else:
                    # NAN here because the step count is 0
                    time_per_step.append(np.nan)

            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(t, time_per_step, "+-")
            plt.xlabel("t")
            plt.ylabel("time_per_step")
            plt.savefig(self.plot_filename + "_time_per_step.png")
            if close_plots:
                plt.close()

            rate = []

            for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
                if sc > 0:
                    rate.append(pc / (td / sc))
                else:
                    # NAN here because the step count is 0
                    rate.append(np.nan)

            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(t, rate, "+-")
            plt.xlabel("t")
            plt.ylabel("Particles / second")
            plt.yscale("log")
            plt.savefig(self.plot_filename + "_rate.png")
            if close_plots:
                plt.close()

            if show_plots:
                plt.show()
