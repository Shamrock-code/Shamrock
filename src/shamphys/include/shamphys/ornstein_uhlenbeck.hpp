// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ornstein_uhlenbeck.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/random/random.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <random>

namespace shammath {

    /////////////////////////////////////////////////////////
    // Standard stuff
    /////////////////////////////////////////////////////////

    /**
     * @brief generate x_n start
     *
     * @param eng A reference to a random number generator engine
     * @param var The variance to scale the generated Gaussian value
     */
    inline f64 init_val_orstein(std::mt19937 &eng, f64 var) {
        return shamalgs::random::mock_gaussian<f64>(eng) * var;
    }

    /**
     * @brief update x_n with Orstein Uhlenbeck
     * \f$ x_n = f x_{n-1} + \sigma \sqrt{1-f^2} z \f$
     */
    template<class Tvec, class Tscal>
    inline void update_orstein(Tvec &x, Tscal f, Tscal sqrt1mf2, Tscal sigma, Tvec z) {
        x = f * x + sigma * sqrt1mf2 * z;
    }

    /////////////////////////////////////////////////////////
    // 6D stuff for solenoidal / compressional forcing
    /////////////////////////////////////////////////////////

    inline f64_3 random_gaussian_vec(std::mt19937 &eng, f64 var) {
        return f64_3{
                   shamalgs::random::mock_gaussian<f64>(eng),
                   shamalgs::random::mock_gaussian<f64>(eng),
                   shamalgs::random::mock_gaussian<f64>(eng)}
               * var;
    }

    /**
     * @brief update x_n with Orstein Uhlenbeck
     * \f$ x_n = f x_{n-1} + \sigma \sqrt{1-f^2} z \f$
     */
    inline void update_orstein_vec(
        std::array<f64_3, 2> &x, f64 f, f64 sqrt1mf2, f64 sigma, std::array<f64_3, 2> z) {
        update_orstein(x[0], f, sqrt1mf2, sigma, z[0]);
        update_orstein(x[1], f, sqrt1mf2, sigma, z[1]);
    }

    /// Shorthand version
    inline void
    update_orstein_vec(std::array<f64_3, 2> &x, f64 f, f64 sigma, std::array<f64_3, 2> z) {
        update_orstein_vec(x, f, sycl::sqrt(1 - f * f), sigma, z);
    }

    /**
     * @brief Generates two random Gaussian vectors
     *
     * @param eng A reference to a random number generator engine
     * @param var The variance to scale the generated Gaussian vectors
     * @return An array containing two `f64_3` random Gaussian vectors
     */
    inline std::array<f64_3, 2> random_gaussian_2vec(std::mt19937 &eng, f64 var) {
        return {random_gaussian_vec(eng, var), random_gaussian_vec(eng, var)};
    }

    /**
     * @brief Convert a random vector to an Orstein Uhlenbeck mode
     *
     * @param x The input vector
     * @param omega The solenoidal weight
     * @param k_unit The unit vector of the mode
     * @return An array containing the two amplitudes
     */
    inline std::array<f64_3, 2>
    orstein_vec_to_mode_ampl(std::array<f64_3, 2> &x, f64 omega, f64_3 k_unit) {

        f64 x_0_dot = sham::dot(x[0], k_unit);
        f64 x_1_dot = sham::dot(x[1], k_unit);

        return {
            omega * (x[0] - k_unit * x_0_dot) + (1 - omega) * k_unit * x_1_dot,
            omega * (x[1] - k_unit * x_1_dot) + (1 - omega) * k_unit * x_0_dot};
    }

} // namespace shammath
