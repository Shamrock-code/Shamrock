// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SedovTaylor.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Sedov-Taylor blast wave analytical solution
 *
 * References:
 * - Sedov, L.I. (1959) "Similarity and Dimensional Methods in Mechanics"
 * - Taylor, G.I. (1950) "The Formation of a Blast Wave by a Very Intense Explosion"
 */

#include "shamphys/SedovTaylor.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

    template<typename T, typename arr_t>
    std::array<size_t, 2> get_closest_range(const arr_t &arr, const T &val, size_t size) {
        size_t low = 0, high = size - 1;

        if (val < arr[low]) {
            return {low, low};
        }

        if (val > arr[high]) {
            return {high, high};
        }

        while (high - low > 1) {

            size_t mid = (low + high) / 2;

            if (arr[mid] < val) {
                low = mid;
            } else {
                high = mid;
            }
        }

        return {low, high};
    }

    template<typename T, typename arr_t>
    T linear_interpolate(const arr_t &arr_x, const arr_t &arr_y, size_t arr_size, const T &x) {

        auto closest_range = get_closest_range(arr_x, x, arr_size);
        size_t left_idx    = closest_range[0];
        size_t right_idx   = closest_range[1];

        if (left_idx == right_idx) {
            return arr_y[left_idx];
        }

        T x0 = arr_x[left_idx];
        T x1 = arr_x[right_idx];
        T y0 = arr_y[left_idx];
        T y1 = arr_y[right_idx];

        if (x1 == x0) {
            return std::numeric_limits<T>::signaling_NaN();
        }

        T interpolated_y = y0 + (x - x0) / (x1 - x0) * (y1 - y0);

        return interpolated_y;
    }

} // anonymous namespace

#include "sedov_soluce_arrays.hpp"

inline size_t ntheoval = sizeof(r_theo) / sizeof(r_theo[0]);

auto shamphys::SedovTaylor::get_value(f64 x) -> field_val {

    f64 rho = linear_interpolate(r_theo, rho_theo, ntheoval, x);
    f64 vx  = linear_interpolate(r_theo, vr_theo, ntheoval, x);
    f64 P   = linear_interpolate(r_theo, p_theo, ntheoval, x);

    return {rho, vx, P};
}

// ============================================================================
// SedovTaylorAnalytical implementation
// ============================================================================

namespace shamphys {

    SedovTaylorAnalytical::SedovTaylorAnalytical(f64 gamma, f64 E_blast, f64 rho_0, u32 ndim)
        : gamma(gamma), E_blast(E_blast), rho_0(rho_0), ndim(ndim), xi_0(compute_xi_0()) {}

    f64 SedovTaylorAnalytical::compute_xi_0() const {
        // Tabulated Sedov constants for common cases
        // These values come from numerical integration of the Sedov equations
        constexpr f64 gamma_5_3 = 5.0 / 3.0;
        constexpr f64 gamma_1_4 = 1.4;
        constexpr f64 tol       = 0.01;

        if (ndim == 3) {
            if (std::abs(gamma - gamma_5_3) < tol) {
                return 1.15167; // 3D, gamma = 5/3 (monatomic gas)
            } else if (std::abs(gamma - gamma_1_4) < tol) {
                return 1.03275; // 3D, gamma = 1.4 (diatomic gas)
            }
        } else if (ndim == 2) {
            if (std::abs(gamma - gamma_1_4) < tol) {
                return 1.033; // 2D, gamma = 1.4
            }
        } else if (ndim == 1) {
            if (std::abs(gamma - gamma_1_4) < tol) {
                return 0.911; // 1D, gamma = 1.4
            }
        }

        // Default fallback - approximate value
        return 1.0;
    }

    f64 SedovTaylorAnalytical::shock_radius(f64 t) const {
        if (t <= 0.0) {
            return 0.0;
        }
        // R_s = xi_0 * (E_blast * t^2 / rho_0)^(1/(ndim+2))
        f64 exponent = 1.0 / static_cast<f64>(ndim + 2);
        return xi_0 * std::pow(E_blast * t * t / rho_0, exponent);
    }

    f64 SedovTaylorAnalytical::shock_velocity(f64 t) const {
        if (t <= 0.0) {
            return 0.0;
        }
        // v_s = dR_s/dt = (2/(ndim+2)) * R_s / t
        f64 R_s = shock_radius(t);
        return 2.0 / static_cast<f64>(ndim + 2) * R_s / t;
    }

    f64 SedovTaylorAnalytical::post_shock_density() const {
        // rho_s = rho_0 * (gamma + 1) / (gamma - 1)
        return rho_0 * (gamma + 1.0) / (gamma - 1.0);
    }

    auto SedovTaylorAnalytical::get_value(f64 r, f64 t) const -> FieldValues {
        constexpr f64 eps = 1e-10;

        // Handle edge cases
        if (t <= eps) {
            return {rho_0, 0.0, eps};
        }

        f64 R_s = shock_radius(t);
        f64 v_s = shock_velocity(t);

        // Outside the shock: ambient conditions
        if (r >= R_s) {
            return {rho_0, 0.0, eps};
        }

        // Similarity variable lambda = r / R_s
        f64 lambda = r / R_s;

        // Post-shock values (Rankine-Hugoniot jump conditions)
        f64 rho_s   = post_shock_density();
        f64 v_shock = 2.0 / (gamma + 1.0) * v_s;               // Post-shock velocity
        f64 P_s     = 2.0 / (gamma + 1.0) * rho_0 * v_s * v_s; // Post-shock pressure

        // Self-similar profiles (approximate)
        // These are simplified approximations of the full Sedov solution
        // For high accuracy, one would need to numerically integrate the ODEs

        // Velocity profile: approximately linear in lambda
        f64 v_r = v_shock * lambda;

        // Density profile: power-law with correction
        // omega = (n+2)*gamma / (2 + n*(gamma-1))
        f64 n     = static_cast<f64>(ndim);
        f64 omega = (n + 2.0) * gamma / (2.0 + n * (gamma - 1.0));

        // Density falls off from shock toward center
        f64 density_factor = std::pow(lambda, omega - 1.0);
        // Correction factor to avoid singularity at center
        f64 correction = std::max(0.1, 1.0 - 0.8 * (1.0 - lambda) * (1.0 - lambda));
        f64 rho        = rho_s * density_factor * correction;

        // Pressure profile: quadratic approximation
        f64 P = P_s * (0.5 + 0.5 * lambda * lambda);

        return {rho, v_r, P};
    }

    void SedovTaylorAnalytical::solution_at_time(
        f64 t,
        const std::vector<f64> &r_values,
        std::vector<f64> &rho_out,
        std::vector<f64> &v_out,
        std::vector<f64> &P_out) const {

        size_t n = r_values.size();
        rho_out.resize(n);
        v_out.resize(n);
        P_out.resize(n);

        for (size_t i = 0; i < n; ++i) {
            FieldValues fv = get_value(r_values[i], t);
            rho_out[i]     = fv.rho;
            v_out[i]       = fv.v_r;
            P_out[i]       = fv.P;
        }
    }

} // namespace shamphys
