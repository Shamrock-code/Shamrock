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
 * @file sets.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <set>
#include <vector>

namespace shambase {

    /**
     * @brief Compute the difference between two containers as three separate vectors.
     *
     * Compute the difference between two containers as three separate vectors.
     * The first vector contains the elements of the first container that are not
     * in the second container. The second vector contains the elements of the
     * second container that are not in the first container. The third vector contains
     * the elements that are present in both containers.
     *
     * @param c1 The first container.
     * @param ref The second container.
     * @param missing The elements of the first container that are not in the second.
     * @param matching The elements that are present in both containers.
     * @param extra The elements of the second container that are not in the first.
     *
     */
    template<class T, class Container1, class Container2>
    inline void set_diff(
        Container1 &c1,
        Container2 &ref,
        std::vector<T> &missing,
        std::vector<T> &matching,
        std::vector<T> &extra) {

        std::set<T> dd_ids;

        for (auto a : c1) {
            dd_ids.insert(a);
        }

        for (auto a : ref) {
            if (dd_ids.find(a) == dd_ids.end()) {
                missing.push_back(a);
            } else {
                matching.push_back(a);
            }
            dd_ids.erase(a);
        }

        for (auto a : dd_ids) {
            extra.push_back(a);
        }
    }
} // namespace shambase
