// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sysinfo.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "sysinfo.hpp"
#include <cpuid.h>

// written from code present on https://en.wikipedia.org/wiki/CPUID
inline std::string get_cpu_name() {
    uint32_t brand[12];

    if (!__get_cpuid_max(0x80000004, NULL)) {
        fprintf(stderr, "Feature not implemented.");
    }

    __get_cpuid(0x80000002, brand + 0x0, brand + 0x1, brand + 0x2, brand + 0x3);
    __get_cpuid(0x80000003, brand + 0x4, brand + 0x5, brand + 0x6, brand + 0x7);
    __get_cpuid(0x80000004, brand + 0x8, brand + 0x9, brand + 0xa, brand + 0xb);
    return shambase::format_printf("Brand: %s\n", brand);
}
