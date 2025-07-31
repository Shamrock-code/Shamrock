// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/typeAliasVec.hpp"
#include "shamsys/Log.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamsys/Log", test_format, 1) {

    std::cout << shambase::format("{} 1", f64_3{0, 1, 2}) << std::endl;
}
