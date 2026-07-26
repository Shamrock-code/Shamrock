cases = {
    "racc=0.8": {
        "path": "_to_trash/circular_disc_sink_False_0.8_10000000",
        "is_torque_free": False,
        "racc": 0.8,
    },
    "racc=0.5": {
        "path": "_to_trash/circular_disc_sink_False_0.5_10000000",
        "is_torque_free": False,
        "racc": 0.5,
    },
    "racc=0.25": {
        "path": "_to_trash/circular_disc_sink_False_0.25_10000000",
        "is_torque_free": False,
        "racc": 0.25,
    },
    "racc=0.8 (torque-free)": {
        "path": "_to_trash/circular_disc_sink_True_0.8_10000000",
        "is_torque_free": True,
        "racc": 0.8,
    },
    "racc=0.5 (torque-free)": {
        "path": "_to_trash/circular_disc_sink_True_0.5_10000000",
        "is_torque_free": True,
        "racc": 0.5,
    },
    "racc=0.25 (torque-free)": {
        "path": "_to_trash/circular_disc_sink_True_0.25_10000000",
        "is_torque_free": True,
        "racc": 0.25,
    },
}

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from shamrock.utils.analysis import (
    AnalysisHelper,
)


def render_rho_profiles(cases):
    helpers = {}
    for key, case in cases.items():
        helpers[key] = {}
        helpers[key]["helper"] = AnalysisHelper(
            analysis_folder=os.path.join(cases[key]["path"], "analysis", "plots"),
            analysis_prefix="density_profile",
        )

    for k in helpers:
        helpers[k]["list_analysis_id"] = helpers[k]["helper"].get_list_analysis_id()

    profile_list_analysis_id1 = helpers["racc=0.8"]["list_analysis_id"]

    for iplot in profile_list_analysis_id1:
        ref_key = None
        data = {}
        for k in helpers:
            if iplot in helpers[k]["list_analysis_id"]:
                data[k] = helpers[k]["helper"].load_analysis(iplot).item()
                if ref_key is None:
                    ref_key = k

        time = data[ref_key]["time"]
        bin_edges_x1d = data[ref_key]["bin_edges_x1d"]
        bin_center = (bin_edges_x1d[:-1] + bin_edges_x1d[1:]) / 2

        plt.figure(dpi=150)
        for k, v in data.items():
            plt.plot(bin_center, v["rho_profile"], label=k)

        text = f"t = {time:0.3f}"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=1)
        plt.gca().add_artist(anchored_text)

        plt.xscale("log")
        plt.yscale("log")
        # plt.ylim(1e-6, 1e-3)
        # plt.xlim(0.1, 21)
        plt.legend(loc="lower left")
        plt.savefig(f"_to_trash/compare_torque_free_sink_rho_profile_{iplot:07}.png")
        plt.close()


def render_Lz_profiles(cases):
    helpers = {}
    for key, case in cases.items():
        helpers[key] = {}
        helpers[key]["helper"] = AnalysisHelper(
            analysis_folder=os.path.join(cases[key]["path"], "analysis", "plots"),
            analysis_prefix="density_profile",
        )

    for k in helpers:
        helpers[k]["list_analysis_id"] = helpers[k]["helper"].get_list_analysis_id()

    profile_list_analysis_id1 = helpers["racc=0.8"]["list_analysis_id"]

    for iplot in profile_list_analysis_id1:
        ref_key = None
        data = {}
        for k in helpers:
            if iplot in helpers[k]["list_analysis_id"]:
                data[k] = helpers[k]["helper"].load_analysis(iplot).item()
                if ref_key is None:
                    ref_key = k

        time = data[ref_key]["time"]
        bin_edges_x1d = data[ref_key]["bin_edges_x1d"]
        bin_center = (bin_edges_x1d[:-1] + bin_edges_x1d[1:]) / 2

        plt.figure(dpi=150)
        for k, v in data.items():
            plt.plot(bin_center, v["Lz_profile"], label=k)

        text = f"t = {time:0.3f}"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=1)
        plt.gca().add_artist(anchored_text)

        plt.xscale("log")
        plt.yscale("log")
        # plt.ylim(1e-6, 1e-3)
        # plt.xlim(0.1, 21)
        plt.legend(loc="lower left")
        plt.savefig(f"_to_trash/compare_torque_free_sink_Lz_profile_{iplot:07}.png")
        plt.close()


def compare_acc_rate(cases):
    curves = {}
    for key, case in cases.items():
        disc_mass_filename = os.path.join(cases[key]["path"], "analysis", "disc_mass.json")

        ret = {}
        with open(disc_mass_filename, "r") as fp:
            disc_mass_data = json.load(fp)["disc_mass"]
            ret["t"] = [d["t"] for d in disc_mass_data]
            ret["disc_mass"] = [d["disc_mass"] for d in disc_mass_data]

            # time derivative of the disc mass
            ret["disc_mass_rate"] = np.diff(ret["disc_mass"]) / np.diff(ret["t"])
        curves[key] = ret

    plt.figure(dpi=150)
    for k, v in curves.items():
        plt.plot(v["t"], v["disc_mass"], label=k)

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-6, 1e-3)
    # plt.xlim(0.1, 21)
    plt.legend(loc="lower left")
    plt.savefig("_to_trash/compare_torque_free_sink_acc_rate_disc_mass.png")
    plt.close()

    plt.figure(dpi=150)
    for k, v in curves.items():
        plt.plot(v["t"][1:], v["disc_mass_rate"], label=k)

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-6, 1e-3)
    # plt.xlim(0.1, 21)
    plt.legend(loc="lower right")
    plt.savefig("_to_trash/compare_torque_free_sink_acc_rate_diff.png")
    plt.close()


def compare_dsink_rate(cases):

    def get_r(pos):
        return np.linalg.norm(pos)

    curves = {}
    for key, case in cases.items():
        disc_mass_filename = os.path.join(cases[key]["path"], "analysis", "sinks.json")

        ret = {}
        with open(disc_mass_filename, "r") as fp:
            sinks_dat = json.load(fp)["sinks"]
            ret["t"] = [d["t"] for d in sinks_dat]
            ret["r"] = [get_r(d["sinks"][0]["pos"]) for d in sinks_dat]

        curves[key] = ret

    plt.figure(dpi=150)
    for k, v in curves.items():
        plt.plot(v["t"], v["r"], label=k)

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-6, 1e-3)
    # plt.xlim(0.1, 21)
    plt.legend(loc="lower left")
    plt.savefig("_to_trash/compare_torque_free_sink_r.png")
    plt.close()


compare_dsink_rate(cases)
compare_acc_rate(cases)
render_rho_profiles(cases)
render_Lz_profiles(cases)
