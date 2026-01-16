// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamcomm/logs.hpp"
#include "shammath/DiscontinuousIterator.hpp"
#include "shamtest/shamtest.hpp"
#include <set>
#include <vector>

inline std::vector<i32> get_vec_result(i32 min, i32 max) {
    std::vector<i32> ret;
    shammath::DiscontinuousIterator<i32> it(min, max);
    while (!it.is_done()) {
        i32 tmp = it.next();
        ret.push_back(tmp);
    }
    return ret;
}

template<typename Set>
inline bool set_compare(Set const &lhs, Set const &rhs) {
    return lhs.size() == rhs.size() && equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool test_range(i32 min, i32 max, std::vector<i32> &table) {
    std::set<i32> ref_set;
    std::set<i32> test_set;

    for (i32 i = min; i < max; i++) {
        ref_set.insert(i);
    }

    for (auto a : table) {
        test_set.insert(a);
    }

    return set_compare(ref_set, test_set);
}

TestStart(Unittest, "shammath/DiscontinuousIterator", iterator, 1) {

    {
        i32 min  = 0;
        i32 max  = 64;
        auto res = get_vec_result(min, max);
        REQUIRE(test_range(min, max, res));

        std::vector<i32> ref_0_64 = {0, 32, 16, 48, 8,  40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60,
                                     2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62,
                                     1, 33, 17, 49, 9,  41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
                                     3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63};

        REQUIRE_EQUAL(res, ref_0_64);
    }
    {
        i32 min  = 0;
        i32 max  = 42;
        auto res = get_vec_result(min, max);
        REQUIRE(test_range(min, max, res));

        std::vector<i32> ref_0_42
            = {0, 32, 16, 8, 40, 24, 4, 36, 20, 12, 28, 2, 34, 18, 10, 26, 6, 38, 22, 14, 30,
               1, 33, 17, 9, 41, 25, 5, 37, 21, 13, 29, 3, 35, 19, 11, 27, 7, 39, 23, 15, 31};

        REQUIRE_EQUAL(res, ref_0_42);
    }
    {
        i32 min  = -65;
        i32 max  = 65;
        auto res = get_vec_result(min, max);
        REQUIRE(test_range(min, max, res));

        std::vector<i32> ref_m65_65
            = {-65, 63,  -1,  -33, 31,  -49, 15,  -17, 47,  -57, 7,   -25, 39,  -41, 23,  -9,  55,
               -61, 3,   -29, 35,  -45, 19,  -13, 51,  -53, 11,  -21, 43,  -37, 27,  -5,  59,  -63,
               1,   -31, 33,  -47, 17,  -15, 49,  -55, 9,   -23, 41,  -39, 25,  -7,  57,  -59, 5,
               -27, 37,  -43, 21,  -11, 53,  -51, 13,  -19, 45,  -35, 29,  -3,  61,  -64, 64,  0,
               -32, 32,  -48, 16,  -16, 48,  -56, 8,   -24, 40,  -40, 24,  -8,  56,  -60, 4,   -28,
               36,  -44, 20,  -12, 52,  -52, 12,  -20, 44,  -36, 28,  -4,  60,  -62, 2,   -30, 34,
               -46, 18,  -14, 50,  -54, 10,  -22, 42,  -38, 26,  -6,  58,  -58, 6,   -26, 38,  -42,
               22,  -10, 54,  -50, 14,  -18, 46,  -34, 30,  -2,  62};

        REQUIRE_EQUAL(res, ref_m65_65);
    }
    {
        i32 min  = -65;
        i32 max  = 0;
        auto res = get_vec_result(min, max);
        REQUIRE(test_range(min, max, res));

        std::vector<i32> ref_m65
            = {-65, -1,  -33, -49, -17, -57, -25, -41, -9,  -61, -29, -45, -13, -53, -21, -37, -5,
               -63, -31, -47, -15, -55, -23, -39, -7,  -59, -27, -43, -11, -51, -19, -35, -3,  -64,
               -32, -48, -16, -56, -24, -40, -8,  -60, -28, -44, -12, -52, -20, -36, -4,  -62, -30,
               -46, -14, -54, -22, -38, -6,  -58, -26, -42, -10, -50, -18, -34, -2};

        REQUIRE_EQUAL(res, ref_m65);
    }
}
