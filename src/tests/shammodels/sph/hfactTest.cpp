// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

inline std::vector<f64> get_h_test_vals() {
    std::vector<f64> ret{};

    for (u32 i = 0; i < 300; i++) {
        f64 hfact = i / 100.;
        ret.push_back(hfact);
    }

    return ret;
}

TestStart(ValidationTest, "shammodels/sph/hfact_default", test_sph_hfact_default, 1) {}
