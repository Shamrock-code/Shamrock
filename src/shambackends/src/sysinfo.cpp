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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/sysinfo.hpp"

// Mac OSX implementation:
#if defined(__MACH__)
    #include <sys/sysctl.h>
    #include <sys/types.h>

std::optional<std::size_t> sham::getPhysicalMemory() {

    int mib[]     = {CTL_HW, HW_MEMSIZE};
    int64_t value = 0;
    size_t length = sizeof(value);

    if (-1 == sysctl(mib, 2, &value, &length, NULL, 0)) {
        return std::nullopt;
    }
    return value;
}

// Linux/BSD implementation:
#elif (defined(linux) || defined(__linux__) || defined(__linux))                                   \
    || (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__)                      \
        || defined(__OpenBSD__))

    #include <sys/sysinfo.h>

std::optional<std::size_t> sham::getPhysicalMemory() {
    struct sysinfo info;
    sysinfo(&info);
    return info.totalram;
}

#else

std::optional<std::size_t> sham::getPhysicalMemory() { return std::nullopt; }

#endif
