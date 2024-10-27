// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// See LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

inline std::vector<f64> get_h_test_vals() {
    std::vector<f64> ret{};

    for (f64 hfact = 0; hfact < 3; hfact += 0.01) {
        ret.push_back(hfact);
    }

    return ret;
}

TestStart(ValidationTest, "shammodels/sph/hfact_default", test_sph_hfact_default, 1) {}
