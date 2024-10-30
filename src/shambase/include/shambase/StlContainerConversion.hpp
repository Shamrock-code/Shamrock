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
 * @file StlContainerConversion.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <set>
#include <vector>

namespace shambase {

    template<class T>
    inline std::vector<T> vector_from_set(const std::set<T> &in) {
        std::vector<T> ret{};
        for (const T &t : in) {
            ret.push_back(t);
        }
        return ret;
    }

    template<class T>
    inline std::set<T> set_from_vector(const std::vector<T> &in) {
        std::set<T> ret{};
        for (const T &t : in) {
            ret.insert(t);
        }
        return ret;
    }

} // namespace shambase
