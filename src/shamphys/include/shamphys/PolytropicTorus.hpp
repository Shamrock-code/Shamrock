// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PolytropicTorus.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Analytic equilibrium of a polytropic torus in (PW) potential
 *
 */

#include "shambase/aliases_float.hpp"
#include <cmath>
#include <utility>

namespace shamphys {

    class PolytropicTorus {
        public:
        /**
         * @brief Build the torus and compute its derived quantities (edges, integrals)
         *
         * @param GM gravitational parameter of the central mass
         * @param r_s pole of the potential, 0 for classical potential, 2GM/c^2 for Paczynski-Wiita
         * @param r0 radius of the pressure (and density) maximum
         * @param r_in inner edge of the torus on the midplane
         * @param a_ang power-law index of the specific angular momentum, in [0, 1/2)
         * @param n_poly polytropic index
         */
        PolytropicTorus(f64 GM, f64 r_s, f64 r0, f64 r_in, f64 a_ang, f64 n_poly);

        /// effective potential (gravity + centrifugal)
        f64 phi_eff(f64 r_cyl, f64 z) const;

        /// gradient (d/dr_cyl, d/dz) of the effective potential
        std::pair<f64, f64> grad_phi_eff(f64 r_cyl, f64 z) const;

        /// W = (n + 1) P / rho, 0 outside of the torus
        f64 enthalpy(f64 r_cyl, f64 z) const;

        /// rho / rho0, 1 at the pressure maximum, 0 outside of the torus
        inline f64 rho_shape(f64 r_cyl, f64 z) const {
            return rho_shape_from_enthalpy(enthalpy(r_cyl, z));
        }

        /// rho_shape for a known W = enthalpy(r_cyl, z)
        inline f64 rho_shape_from_enthalpy(f64 W) const { return std::pow(W / W0, n_poly); }

        /// gradient (d/dr_cyl, d/dz) of rho_shape
        std::pair<f64, f64> grad_rho_shape(f64 r_cyl, f64 z) const;

        /// P / rho = W / (n + 1)
        inline f64 p_over_rho(f64 r_cyl, f64 z) const {
            return p_over_rho_from_enthalpy(enthalpy(r_cyl, z));
        }

        /// p_over_rho for a known W = enthalpy(r_cyl, z)
        inline f64 p_over_rho_from_enthalpy(f64 W) const { return W / (n_poly + 1); }

        /// azimuthal velocity l / r_cyl
        f64 v_phi(f64 r_cyl) const;

        /**
         * @brief Poloidal field (B_r_cyl, B_z) = curl(A_phi e_phi)
         */
        std::pair<f64, f64> b_poloidal_unit(f64 r_cyl, f64 z, f64 A_cut) const;

        /// b_poloidal_unit for a known W = enthalpy(r_cyl, z)
        std::pair<f64, f64> b_poloidal_unit_from_enthalpy(f64 r_cyl, f64 z, f64 W, f64 A_cut) const;

        /// integral of rho_shape over the torus volume (torus mass for rho0 = 1)
        f64 volume_integral() const { return vol_integral; }

        /// integral of P over the torus volume for rho0 = 1
        f64 pressure_integral() const { return p_integral; }

        /// integral of |b_poloidal_unit|^2 over the torus volume
        f64 b_poloidal_unit_sq_integral(f64 A_cut) const;

        f64 get_r0() const { return r0; }
        f64 get_r_in() const { return r_in; }

        /// outer edge of the torus on the midplane
        f64 get_r_out() const { return r_out; }
        /// maximum height of the torus surface
        f64 get_z_max() const { return z_max; }
        /// radius of the cusp of the effective potential (0 if there is none)
        f64 get_r_cusp() const { return r_cusp; }
        /// L0 such that l = L0 r_cyl^a
        f64 get_L0() const { return L0; }
        /// W at the pressure maximum
        f64 get_W0() const { return W0; }

        private:
        /// grad_rho_shape for a known W = enthalpy(r_cyl, z)
        std::pair<f64, f64> grad_rho_shape_from_enthalpy(f64 r_cyl, f64 z, f64 W) const;

        f64 GM, r_s, r0, r_in, a_ang, n_poly;

        f64 L0, phi_eff_0, W0;
        f64 r_cusp, r_out, z_max;
        f64 vol_integral, p_integral;

        f64 find_r_cusp() const;
        f64 find_r_out() const;
        f64 find_z_surface(f64 r_cyl) const;
        f64 find_z_max() const;

        /// midpoint quadrature over the torus volume of f(r_cyl, z) (z >= 0 half, x2)
        template<class Func>
        f64 integrate(Func &&f) const;
    };

} // namespace shamphys
