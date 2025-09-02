// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file mock_vector.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Utility functions for generating random mock vectors
 *
 * This header provides template functions for generating random mock vectors of various
 * types, including primitive types (integers, floats) and SYCL vectors. These functions
 * are useful for generating sample data collections.
 */

#include "shamalgs/details/random/random.hpp"
#include <random>
#include <vector>

namespace shamalgs::primitives {

    /**
     * @brief Generates a vector of random mock values within specified bounds
     *
     * This function generates a std::vector of random values of type T within the range
     * [min_bound, max_bound]. Each element in the vector is generated independently using
     * the same bounds. The function uses a deterministic random number generator seeded
     * with the provided seed value, ensuring reproducible results for the same seed.
     *
     * @tparam T The type of values to generate (supports primitive types and SYCL vectors)
     * @param seed The seed value for the random number generator
     * @param len The length of the vector to generate
     * @param min_bound The minimum bound for the generated values
     * @param max_bound The maximum bound for the generated values
     * @return A std::vector<T> containing len random values within the specified bounds
     *
     * @code{.cpp}
     * #include "shamalgs/primitives/mock_vector.hpp"
     *
     * // Generate a vector of random integers
     * std::vector<i32> random_ints = shamalgs::mock_vector<i32>(42, 10, i32{0}, i32{100});
     *
     * // Generate a vector of random floats
     * std::vector<f32> random_floats = shamalgs::mock_vector<f32>(123, 5, f32{0.0f}, f32{1.0f});
     *
     * // Generate a vector of random SYCL vectors
     * std::vector<sycl::vec<f64, 3>> random_vecs = shamalgs::mock_vector<sycl::vec<f64, 3>>(
     *     456,
     *     3,
     *     sycl::vec<f64, 3>{0.0, 0.0, 0.0},
     *     sycl::vec<f64, 3>{1.0, 1.0, 1.0}
     * );
     * @endcode
     */
    template<class T>
    std::vector<T> mock_vector(u64 seed, u32 len, T min_bound, T max_bound) {
        std::vector<T> vec;
        vec.reserve(len);

        std::mt19937 eng(seed);

        for (u32 i = 0; i < len; i++) {
            vec.push_back(mock_value(eng, min_bound, max_bound));
        }

        return vec;
    }

    /**
     * @brief Generates a vector of random mock values using default bounds
     *
     * This function generates a std::vector of random values of type T using the default
     * minimum and maximum bounds defined by shambase::VectorProperties<T>. This is a
     * convenience function that automatically determines appropriate bounds based on the
     * type T. The function uses a deterministic random number generator seeded with the
     * provided seed value.
     *
     * @tparam T The type of values to generate (supports primitive types and SYCL vectors)
     * @param seed The seed value for the random number generator
     * @param len The length of the vector to generate
     * @return A std::vector<T> containing len random values within the default bounds for the type
     *
     * @code{.cpp}
     * #include "shamalgs/primitives/mock_vector.hpp"
     *
     * // Generate vectors with default bounds
     * std::vector<i32> random_ints = shamalgs::mock_vector<i32>(42, 10);
     * std::vector<f64> random_doubles = shamalgs::mock_vector<f64>(123, 5);
     * std::vector<sycl::vec<f32, 2>> random_vecs = shamalgs::mock_vector<sycl::vec<f32, 2>>(456,
     * 3);
     * @endcode
     */
    template<class T>
    inline std::vector<T> mock_vector(u64 seed, u32 len) {
        using Prop = shambase::VectorProperties<T>;
        return mock_vector(seed, len, Prop::get_min(), Prop::get_max());
    }

} // namespace shamalgs::primitives
