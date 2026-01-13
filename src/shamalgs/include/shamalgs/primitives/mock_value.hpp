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
 * @file mock_value.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Utility functions for generating random mock values
 *
 * This header provides template functions for generating random mock values of various
 * types, including primitive types (integers, floats) and SYCL vectors. These functions
 * are useful for generating sample data.
 */

#include "shambackends/vec.hpp"
#include <random>

namespace shamalgs::primitives {

    /**
     * @brief Generates a random mock value within specified bounds
     *
     * This function generates a random value of type T within the range [min_bound, max_bound].
     * For integer types, the distribution is uniform over the inclusive range.
     * For floating-point types, the distribution is uniform over the range [min_bound, max_bound).
     * For SYCL vectors, each component is generated independently using the same bounds.
     *
     * @tparam T The type of value to generate (supports primitive types and SYCL vectors)
     * @param eng The random number generator engine to use
     * @param min_bound The minimum bound for the generated value
     * @param max_bound The maximum bound for the generated value
     * @return A random value of type T within the specified bounds
     *
     * @code{.cpp}
     * #include <random>
     * #include "shamalgs/primitives/mock_value.hpp"
     *
     * // Initialize random number generator
     * std::mt19937 rng(42);
     *
     * // Generate random integers
     * i32 random_int = shamalgs::mock_value(rng, i32{0}, i32{100});
     *
     * // Generate random floats
     * f32 random_float = shamalgs::mock_value(rng, f32{0.0f}, f32{1.0f});
     *
     * // Generate random SYCL vectors
     * sycl::vec<f32, 3> random_vec = shamalgs::mock_value(
     *     rng,
     *     sycl::vec<f32, 3>{0.0f, 0.0f, 0.0f},
     *     sycl::vec<f32, 3>{1.0f, 1.0f, 1.0f}
     * );
     * @endcode
     */
    template<class T>
    T mock_value(std::mt19937 &eng, T min_bound, T max_bound);

    /**
     * @brief Generates a random mock value using default bounds
     *
     * This function generates a random value of type T using the default minimum and maximum
     * bounds defined by shambase::VectorProperties<T>. This is a convenience function that
     * automatically determines appropriate bounds based on the type T.
     *
     * @tparam T The type of value to generate (supports primitive types and SYCL vectors)
     * @param eng The random number generator engine to use
     * @return A random value of type T within the default bounds for the type
     *
     * @code{.cpp}
     * #include <random>
     * #include "shamalgs/primitives/mock_value.hpp"
     *
     * // Initialize random number generator
     * std::mt19937 rng(42);
     *
     * // Generate random values with default bounds
     * i32 random_int = shamalgs::mock_value<i32>(rng);
     * f64 random_double = shamalgs::mock_value<f64>(rng);
     * sycl::vec<f32, 2> random_vec2 = shamalgs::mock_value<sycl::vec<f32, 2>>(rng);
     * @endcode
     */
    template<class T>
    inline T mock_value(std::mt19937 &eng) {
        using Prop = shambase::VectorProperties<T>;
        return mock_value<T>(eng, Prop::get_min(), Prop::get_max());
    }

} // namespace shamalgs::primitives
