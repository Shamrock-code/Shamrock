// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PolytropicTorus.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include "shamcomm/logs.hpp"
#include "shamphys/PolytropicTorus.hpp"
#include <cmath>
#include <stdexcept>

namespace {

    // Should move that in shamath or something later on
    template<class Func>
    f64 golden_section_max(Func &&f, f64 a, f64 b, u32 niter_max = 200) {
        const f64 inv_golden_nb = (std::sqrt(5.) - 1) / 2;
        f64 c                   = b - inv_golden_nb * (b - a);
        f64 d                   = a + inv_golden_nb * (b - a);
        f64 fc                  = f(c);
        f64 fd                  = f(d);
        for (u32 i = 0; i < niter_max; i++) {
            if (fc > fd) {
                b  = d;
                d  = c;
                fd = fc;
                c  = b - inv_golden_nb * (b - a);
                fc = f(c);
            } else {
                a  = c;
                c  = d;
                fc = fd;
                d  = a + inv_golden_nb * (b - a);
                fd = f(d);
            }
        }
        return (a + b) / 2;
    }

    template<class Func>
    f64 bisect(Func &&f, f64 a, f64 b, u32 niter_max = 200) {
        f64 fa = f(a);
        for (u32 i = 0; i < niter_max; i++) {
            f64 m  = (a + b) / 2;
            f64 fm = f(m);
            if ((fm > 0) == (fa > 0)) {
                a  = m;
                fa = fm;
            } else {
                b = m;
            }
        }
        return (a + b) / 2;
    }

} // namespace

namespace shamphys {

    PolytropicTorus::PolytropicTorus(f64 GM, f64 r_s, f64 r0, f64 r_in, f64 a_ang, f64 n_poly)
        : GM(GM), r_s(r_s), r0(r0), r_in(r_in), a_ang(a_ang), n_poly(n_poly) {

        if (!(n_poly > 0)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("PolytropicTorus: n_poly must be > 0, got {}", n_poly));
        }
        if (!(a_ang >= 0 && a_ang < 0.5)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("PolytropicTorus: a_ang must be in [0, 1/2), got {}", a_ang));
        }
        if (!(r0 > r_s)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("PolytropicTorus: r0 = {} must be larger than r_s = {}", r0, r_s));
        }

        f64 dln_lk2
            = 3
              - 2 * r0 / (r0 - r_s); // logarithmic slope of the squared Keplerian angular momentum

        if (!(dln_lk2 > 2 * a_ang)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "PolytropicTorus: no pressure maximum at r0 = {} (r_s = {}, a = {}), r0 is too "
                "close to the "
                "central mass",
                r0,
                r_s,
                a_ang));
        }

        f64 l0_sq = GM * r0 * r0 * r0 / ((r0 - r_s) * (r0 - r_s));
        L0        = std::sqrt(l0_sq) / std::pow(r0, a_ang);

        r_cusp = find_r_cusp();
        if (!(r_in > r_cusp && r_in < r0)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "PolytropicTorus: r_in = {} must be in ]r_cusp, r0[ = ]{}, {}[, otherwise the "
                "torus overflows its Roche lobe or has no pressure maximum",
                r_in,
                r_cusp,
                r0));
        }

        phi_eff_0 = phi_eff(r_in, 0);
        if (!(phi_eff_0 < 0)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "PolytropicTorus: the torus is not bound (Phi_eff(r_in) = {} >= 0), increase r_in",
                phi_eff_0));
        }

        W0 = phi_eff_0 - phi_eff(r0, 0);

        r_out = find_r_out();
        z_max = find_z_max();

        vol_integral = integrate([&](f64 r_cyl, f64 z) {
            return rho_shape(r_cyl, z);
        });
        p_integral   = integrate([&](f64 r_cyl, f64 z) {
            return rho_shape(r_cyl, z) * p_over_rho(r_cyl, z);
        });
    }

    f64 PolytropicTorus::phi_eff(f64 r_cyl, f64 z) const {
        f64 r = std::sqrt(r_cyl * r_cyl + z * z);
        return -GM / (r - r_s) + L0 * L0 * std::pow(r_cyl, 2 * a_ang - 2) / (2 - 2 * a_ang);
    }

    std::pair<f64, f64> PolytropicTorus::grad_phi_eff(f64 r_cyl, f64 z) const {
        f64 r       = std::sqrt(r_cyl * r_cyl + z * z);
        f64 dphi_dr = GM / ((r - r_s) * (r - r_s));
        return {dphi_dr * r_cyl / r - L0 * L0 * std::pow(r_cyl, 2 * a_ang - 3), dphi_dr * z / r};
    }

    f64 PolytropicTorus::enthalpy(f64 r_cyl, f64 z) const {
        // the region inside the cusp is not connected to the torus
        if (r_cyl <= r_cusp) {
            return 0;
        }
        f64 W = phi_eff_0 - phi_eff(r_cyl, z);
        return (W > 0) ? W : 0; // no negative density
    }

    std::pair<f64, f64> PolytropicTorus::grad_rho_shape(f64 r_cyl, f64 z) const {
        return grad_rho_shape_from_enthalpy(r_cyl, z, enthalpy(r_cyl, z));
    }

    std::pair<f64, f64> PolytropicTorus::grad_rho_shape_from_enthalpy(
        f64 r_cyl, f64 z, f64 W) const {
        if (W <= 0) {
            return {0, 0};
        }
        // d rho_shape = n (W / W0)^(n-1) dW / W0, with dW = -d Phi_eff
        f64 fact                  = n_poly * std::pow(W / W0, n_poly - 1) / W0;
        auto [dphi_r_cyl, dphi_z] = grad_phi_eff(r_cyl, z);
        return {-fact * dphi_r_cyl, -fact * dphi_z};
    }

    f64 PolytropicTorus::v_phi(f64 r_cyl) const { return L0 * std::pow(r_cyl, a_ang - 1); }

    std::pair<f64, f64> PolytropicTorus::b_poloidal_unit(f64 r_cyl, f64 z, f64 A_cut) const {
        return b_poloidal_unit_from_enthalpy(r_cyl, z, enthalpy(r_cyl, z), A_cut);
    }

    std::pair<f64, f64> PolytropicTorus::b_poloidal_unit_from_enthalpy(
        f64 r_cyl, f64 z, f64 W, f64 A_cut) const {
        f64 A = rho_shape_from_enthalpy(W) - A_cut;
        if (A <= 0) {
            return {0, 0};
        }
        // B_varpi = -dA/dz, B_z = (1/varpi) d(varpi A)/dvarpi = A / varpi + dA/dvarpi
        auto [drho_r_cyl, drho_z] = grad_rho_shape_from_enthalpy(r_cyl, z, W);
        return {-drho_z, A / r_cyl + drho_r_cyl};
    }

    f64 PolytropicTorus::b_poloidal_unit_sq_integral(f64 A_cut) const {
        return integrate([&](f64 r_cyl, f64 z) {
            auto [B_r_cyl, B_z] = b_poloidal_unit(r_cyl, z, A_cut);
            return B_r_cyl * B_r_cyl + B_z * B_z;
        });
    }

    f64 PolytropicTorus::find_r_cusp() const {
        if (r_s == 0) {
            return 0;
        }
        // Phi_eff -> -inf at r_s
        f64 eps = 1e-9 * r_s;
        return golden_section_max(
            [&](f64 r_cyl) {
                return phi_eff(r_cyl, 0);
            },
            r_s + eps,
            r0);
    }

    f64 PolytropicTorus::find_r_out() const {
        // r_out is where the enthalpy goes to 0
        //  W -> phi_eff_0 < 0 when r -> rout
        auto W = [&](f64 r_cyl) {
            return phi_eff_0 - phi_eff(r_cyl, 0);
        };

        // find an interval where W become negative
        f64 hi = 2 * r0;
        while (W(hi) >= 0) {
            hi *= 2;
        }

        // find the zero
        return bisect(W, r0, hi);
    }

    f64 PolytropicTorus::find_z_surface(f64 r_cyl) const {
        // fidn the height of the surface

        auto W = [&](f64 z) {
            return phi_eff_0 - phi_eff(r_cyl, z);
        };

        if (W(0) <= 0) { // torus is flat
            return 0;
        }

        f64 hi = r0;
        while (W(hi) >= 0) {
            hi *= 2;
        }
        return bisect(W, 0, hi);
    }

    f64 PolytropicTorus::find_z_max() const {
        // maz height of the torus

        // rougly find the max
        const u32 N = 256;
        f64 dr_cyl  = (r_out - r_in) / N;
        u32 i_best  = 0;
        f64 z_best  = 0;
        for (u32 i = 0; i <= N; i++) {
            f64 z = find_z_surface(r_in + i * dr_cyl);
            if (z > z_best) {
                z_best = z;
                i_best = i;
            }
        }

        // find precise location through golden search
        f64 lo         = r_in + std::max<i64>(i64(i_best) - 1, 0) * dr_cyl;
        f64 hi         = r_in + std::min<u32>(i_best + 1, N) * dr_cyl;
        f64 r_cyl_best = golden_section_max(
            [&](f64 r_cyl) {
                return find_z_surface(r_cyl);
            },
            lo,
            hi);
        return std::max(z_best, find_z_surface(r_cyl_best));
    }

    template<class Func>
    f64 PolytropicTorus::integrate(Func &&f) const {
        const u32 nr_cyl = 1024;
        const u32 nz     = 512;
        f64 dr_cyl       = (r_out - r_in) / nr_cyl;
        f64 dz           = z_max / nz;
        f64 sum          = 0;
        for (u32 i = 0; i < nr_cyl; i++) {
            f64 r_cyl = r_in + (i + 0.5) * dr_cyl;
            for (u32 j = 0; j < nz; j++) {
                f64 z = (j + 0.5) * dz;
                sum += f(r_cyl, z) * r_cyl;
            }
        }
        // 2 pi varpi dvarpi dz, times 2 for z < 0
        return 2 * 2 * shambase::constants::pi<f64> * sum * dr_cyl * dz;
    }

} // namespace shamphys
