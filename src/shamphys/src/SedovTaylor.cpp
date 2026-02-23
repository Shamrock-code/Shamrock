// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SedovTaylor.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sod tube analytical solution adapted from a script of Leodasce Sewanou
 */

#include "shamphys/SedovTaylor.hpp"
#include <array>
#include <cstddef>
#include <limits>

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

#include "sedov_soluce_arrays.hpp"

inline size_t ntheoval = sizeof(r_theo) / sizeof(r_theo[0]);

auto shamphys::SedovTaylor::get_value(f64 x) -> field_val {

    f64 rho = linear_interpolate(r_theo, rho_theo, ntheoval, x);
    f64 vx  = linear_interpolate(r_theo, vr_theo, ntheoval, x);
    f64 P   = linear_interpolate(r_theo, p_theo, ntheoval, x);

    return {rho, vx, P};
}
