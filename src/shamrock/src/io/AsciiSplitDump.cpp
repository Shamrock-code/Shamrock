// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AsciiSplitDump.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/io/AsciiSplitDump.hpp"

template<class T>
void AsciiSplitDump::PatchDump::write_val(T val) {
    if constexpr (
        std::is_same_v<T, u32> || std::is_same_v<T, i32> || std::is_same_v<T, u64>
        || std::is_same_v<T, i64>) {
        file << shambase::format("{:}\n", val);
    } else if constexpr (std::is_same_v<T, i64_3> || std::is_same_v<T, i32_3>) {
        file << shambase::format("{:} {:} {:}\n", val.x(), val.y(), val.z());
    } else if constexpr (std::is_same_v<T, f64> || std::is_same_v<T, f32>) {
        file << shambase::format("{:0.9f}\n", val);
    } else if constexpr (std::is_same_v<T, f64_3> || std::is_same_v<T, f32_3>) {
        file << shambase::format("{:0.9f} {:0.9f} {:0.9f}\n", val.x(), val.y(), val.z());
    } else if constexpr (std::is_same_v<T, f64_8> || std::is_same_v<T, f32_8>) {
        file << shambase::format(
            "{:0.9f} {:0.9f} {:0.9f} {:0.9f} {:0.9f} {:0.9f} {:0.9f} {:0.9f}\n",
            val.s0(),
            val.s1(),
            val.s2(),
            val.s3(),
            val.s4(),
            val.s5(),
            val.s6(),
            val.s7());
    } else {
        shambase::throw_unimplemented();
    }
}

template<class T>
void AsciiSplitDump::PatchDump::write_table(std::vector<T> buf, u32 len) {
    for (u32 i = 0; i < len; i++) {
        write_val(buf[i]);
    }
}

template<class T>
void AsciiSplitDump::PatchDump::write_table(sycl::buffer<T> buf, u32 len) {
    sycl::host_accessor acc{buf, sycl::read_only};
    for (u32 i = 0; i < len; i++) {
        write_val(acc[i]);
    }
}

#ifndef DOXYGEN
    #define XMAC_TYPES                                                                             \
        X(f32)                                                                                     \
        X(f32_2)                                                                                   \
        X(f32_3)                                                                                   \
        X(f32_4)                                                                                   \
        X(f32_8)                                                                                   \
        X(f32_16)                                                                                  \
        X(f64)                                                                                     \
        X(f64_2)                                                                                   \
        X(f64_3)                                                                                   \
        X(f64_4)                                                                                   \
        X(f64_8)                                                                                   \
        X(f64_16)                                                                                  \
        X(u32)                                                                                     \
        X(u64)                                                                                     \
        X(u32_3)                                                                                   \
        X(u64_3)                                                                                   \
        X(i64_3)

    #define X(_arg_)                                                                               \
        template void AsciiSplitDump::PatchDump::write_val<_arg_>(_arg_ val);                      \
        template void AsciiSplitDump::PatchDump::write_table<_arg_>(                               \
            std::vector<_arg_> buf, u32 len);                                                      \
        template void AsciiSplitDump::PatchDump::write_table<_arg_>(                               \
            sycl::buffer<_arg_> buf, u32 len);

XMAC_TYPES
    #undef X
#endif
