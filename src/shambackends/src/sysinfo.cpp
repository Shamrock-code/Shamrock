// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sysinfo.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sysinfo.hpp"

// Mac OSX implementation:
#if defined(__MACH__)
    #include <mach/mach.h>
    #include <sys/sysctl.h>
    #include <sys/types.h>

std::optional<std::size_t> sham::getHostPhysicalMemory() {

    int mib[]     = {CTL_HW, HW_MEMSIZE};
    int64_t value = 0;
    size_t length = sizeof(value);

    if (-1 == sysctl(mib, 2, &value, &length, NULL, 0)) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::size_t> sham::getHostAvailableMemory() {

    vm_size_t page_size = 0;
    if (KERN_SUCCESS != host_page_size(mach_host_self(), &page_size)) {
        return std::nullopt;
    }

    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (KERN_SUCCESS
        != host_statistics64(
            mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stats), &count)) {
        return std::nullopt;
    }

    // Free memory plus reclaimable (inactive) pages, as reported as "available" by most tools.
    std::size_t available_pages = static_cast<std::size_t>(vm_stats.free_count)
                                  + static_cast<std::size_t>(vm_stats.inactive_count);

    return available_pages * static_cast<std::size_t>(page_size);
}

// Linux/BSD implementation:
#elif (defined(linux) || defined(__linux__) || defined(__linux))                                   \
    || (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__)                      \
        || defined(__OpenBSD__))

    #include <sys/sysinfo.h>
    #include <fstream>
    #include <string>

std::optional<std::size_t> sham::getHostPhysicalMemory() {
    struct sysinfo info;
    sysinfo(&info);
    return info.totalram;
}

std::optional<std::size_t> sham::getHostAvailableMemory() {
    // Prefer /proc/meminfo's MemAvailable, which accounts for reclaimable caches/buffers
    // (unlike sysinfo's freeram field).
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo) {
        std::string label;
        std::size_t value_kb;
        std::string unit;
        while (meminfo >> label >> value_kb >> unit) {
            if (label == "MemAvailable:") {
                return value_kb * 1024;
            }
        }
    }

    struct sysinfo info;
    if (-1 == sysinfo(&info)) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(info.freeram) * static_cast<std::size_t>(info.mem_unit);
}

#else

std::optional<std::size_t> sham::getHostPhysicalMemory() { return std::nullopt; }
std::optional<std::size_t> sham::getHostAvailableMemory() { return std::nullopt; }

#endif
