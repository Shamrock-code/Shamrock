import numpy as np

import shamrock


def rank_0_print(*args, **kwargs):
    if shamrock.sys.world_rank() == 0:
        print(*args, **kwargs)


class DustMRNDistribution:
    def __init__(self, codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges):
        self.codeu = codeu
        self.mrn_pow = mrn_pow
        self.mrn_cutoff_si = mrn_cutoff_si
        self.grain_size_si_edges = grain_size_si_edges
        self.rho_grains_si_edges = rho_grains_si_edges

        to_m = self.codeu.get("m")
        to_kg = self.codeu.get("kg")
        to_dens = to_kg * to_m**-3

        rank_0_print(" ---- MRN DISTRIBUTION ----")

        rank_0_print("bin edges:")
        rank_0_print(f"  grains sizes (edges) = {self.grain_size_si_edges.tolist()} [m]")
        rank_0_print(f"  grains dens (edges) = {self.rho_grains_si_edges.tolist()} [kg.m^-3]")

        self.grain_size_edges = self.grain_size_si_edges * to_m
        self.rho_grains_edges = to_dens * np.array(self.rho_grains_si_edges)

        rank_0_print(f"  grains sizes (edges) = {self.grain_size_edges.tolist()} [code u]")
        rank_0_print(f"  grains dens (edges) = {self.rho_grains_edges.tolist()} [code u]")

        rank_0_print()
        rank_0_print("bin centers (geom averages):")

        self.grain_size = np.sqrt(self.grain_size_edges[:-1] * self.grain_size_edges[1:])
        self.rho_grains = np.sqrt(self.rho_grains_edges[:-1] * self.rho_grains_edges[1:])

        self.bin_width_si = self.grain_size_si_edges[1:] - self.grain_size_si_edges[:-1]
        self.bin_width = self.grain_size_edges[1:] - self.grain_size_edges[:-1]

        self.grain_size_si = np.sqrt(self.grain_size_si_edges[:-1] * self.grain_size_si_edges[1:])
        self.rho_grains_si = np.sqrt(self.rho_grains_si_edges[:-1] * self.rho_grains_si_edges[1:])

        rank_0_print(f"  grains sizes (bin) = {self.grain_size_si.tolist()} [m]")
        rank_0_print(f"  grains dens (bin) = {self.rho_grains_si.tolist()} [kg.m^-3]")

        rank_0_print(f"  grains sizes (bin) = {self.grain_size.tolist()} [code units]")
        rank_0_print(f"  grains dens (bin) = {self.rho_grains.tolist()} [code units]")

        self.massgrid_edges = (4 * np.pi / 3) * self.rho_grains_edges * self.grain_size_edges**3
        self.massgrid = np.sqrt(self.massgrid_edges[:-1] * self.massgrid_edges[1:])

        self.massgrid_si_edges = self.massgrid_edges * self.codeu.to("kg")
        self.massgrid_si = self.massgrid * self.codeu.to("kg")

        rank_0_print(f"  massgrid = {self.massgrid_si.tolist()} [kg]")
        rank_0_print(f"  massgrid = {self.massgrid.tolist()} [code units]")

        rank_0_print()
        rank_0_print("deduced:")

        self.alpha = 3 - self.mrn_pow
        rank_0_print(f"  alpha = 3 - mrn_pow = {self.alpha}")

        max_s = min(self.mrn_cutoff_si, max(self.grain_size_si_edges))
        min_s = min(self.mrn_cutoff_si, min(self.grain_size_si_edges))
        rank_0_print(f"  s (edges) max = {max_s} min = {min_s} [m]")

        self.grain_size_si_edges_clipped = np.clip(
            self.grain_size_si_edges, None, self.mrn_cutoff_si
        )
        self.grain_size_si_clipped = np.clip(self.grain_size_si, None, self.mrn_cutoff_si)
        rank_0_print(
            f"  grains sizes (edges) (clipped) = {self.grain_size_si_edges_clipped.tolist()} [m]"
        )

        # d espilon = epsilon_0 s^\alpha d s
        # \int d epsilon_s = [ s^(alpha + 1) / (alpha + 1) ]^max_min

        def prim(s):
            return s ** (self.alpha + 1) / (self.alpha + 1)

        self.mrn_weight = prim(self.grain_size_si_edges_clipped[1:]) - prim(
            self.grain_size_si_edges_clipped[:-1]
        )
        self.mrn_weight = self.mrn_weight / np.sum(self.mrn_weight)  # normalize to 1
        rank_0_print(f"  mrn_weight = {self.mrn_weight.tolist()}")
        rank_0_print(f"  sum(mrn_weight) = {np.sum(self.mrn_weight)}")

        # here sum(mrn_weight) = \int d epsilon = 1 by design
        # S_mean = \int s depsilon / \int depsilon
        #    = (analytics) [ \epsilon_0 s^(alpha + 2) / (alpha + 2) ]^max_min
        #                / [ \epsilon_0 s^(alpha + 1) / (alpha + 1) ]^max_min
        #    (numerically)
        # average epsilon_j = \int d epsilon / \delta_bin

        def s_mean_pow_1(add_pow, val):
            return val ** (self.alpha + add_pow) / (self.alpha + add_pow)

        def s_mean_pow_2(add_pow):
            return s_mean_pow_1(add_pow, max_s) - s_mean_pow_1(add_pow, min_s)

        analytical_S_mean = s_mean_pow_2(2) / s_mean_pow_2(1)

        S_mean = np.sum(self.mrn_weight * self.grain_size_si)
        rank_0_print(
            f"  S_mean = {S_mean} S_mean_init = {analytical_S_mean} (diff = {S_mean - analytical_S_mean})"
        )

        rank_0_print(" -------------------------")
