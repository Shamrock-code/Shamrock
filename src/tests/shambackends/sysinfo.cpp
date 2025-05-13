// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/sysinfo.hpp"
#include "shambase/string.hpp"
#include "fmt/std.h"
#include "shamcomm/logs.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambackends/sysinfo:getPhysicalMemory", test_sysinfo_getPhysicalMemory, 1) {
    auto phys_mem = sham::getPhysicalMemory();

    logger::raw_ln("Physical memory: bool(result)", bool(phys_mem));
    if (phys_mem) {
        logger::raw_ln("Physical memory: size =", shambase::readable_sizeof(*phys_mem));
    }

    REQUIRE(bool(phys_mem));
}
