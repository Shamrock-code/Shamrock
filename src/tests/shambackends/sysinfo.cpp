// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/sysinfo.hpp"
#include "shambase/string.hpp"
#include "fmt/std.h"
#include "shamcomm/logs.hpp"
#include "shamtest/shamtest.hpp"

NEW_TEST(Unittest, "shambackends/sysinfo:getHostPhysicalMemory", 1) {
    auto phys_mem = sham::getHostPhysicalMemory();

    logger::raw_ln("Physical memory: bool(result)", bool(phys_mem));
    if (phys_mem) {
        logger::raw_ln("Physical memory: size =", shambase::readable_sizeof(*phys_mem));
    }

    REQUIRE(bool(phys_mem));
}

NEW_TEST(Unittest, "shambackends/sysinfo:getHostAvailableMemory", 1) {
    auto avail_mem = sham::getHostAvailableMemory();

    logger::raw_ln("Available memory: bool(result)", bool(avail_mem));
    if (avail_mem) {
        logger::raw_ln("Available memory: size =", shambase::readable_sizeof(*avail_mem));
    }

    REQUIRE(bool(avail_mem));
}
