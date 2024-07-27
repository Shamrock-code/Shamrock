// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
