// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/sets.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

TestStart(Unittest, "shambase/sets::set_diff", set_diff_test, 1) {

    std::vector<int> v1{0, 1, 3, 4, 5};
    std::vector<int> ref{0, 1, 2, 8, 4, 5};

    std::vector<int> missing;
    std::vector<int> matching;
    std::vector<int> extra;

    shambase::set_diff(v1, ref, missing, matching, extra);

    std::vector<int> ref_missing{2, 8};
    std::vector<int> ref_matching{0, 1, 4, 5};
    std::vector<int> ref_extra{3};

    REQUIRE_EQUAL(missing, ref_missing);
    REQUIRE_EQUAL(matching, ref_matching);
    REQUIRE_EQUAL(extra, ref_extra);
}
