// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shammath/DiscontinuousIterator.hpp"
#include "shamtest/shamtest.hpp"
#include <set>

template<typename Set>
inline bool set_compare(Set const &lhs, Set const &rhs) {
    return lhs.size() == rhs.size() && equal(lhs.begin(), lhs.end(), rhs.begin());
}

void test_range(i32 min, i32 max) {
    std::set<i32> ref_set;
    std::set<i32> test_set;

    for (i32 i = min; i < max; i++) {
        ref_set.insert(i);
    }

    shammath::DiscontinuousIterator<i32> it(min, max);
    while (!it.is_done()) {
        i32 tmp = it.next();
        test_set.insert(tmp);
    }

    REQUIRE(set_compare(ref_set, test_set));
}

TestStart(Unittest, "shammath/DiscontinuousIterator", iterator, 1) {

    test_range(0, 64);
    test_range(0, 42);
    test_range(-65, 65);
    test_range(-65, 0);
}
