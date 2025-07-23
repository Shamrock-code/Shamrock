// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file bmi.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Bit manipulation instruction implementation for SYCL
 * @version 0.1
 * @date 2022-03-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "shambase/aliases_int.hpp"

namespace shamrock::sfc::bmi {

    template<class inttype, int interleaving>
    inttype expand_bits(inttype);

    // 21bit number and 2 interleaving bits (for 64bits)
    template<>
    inline u64 expand_bits<u64, 2>(u64 x) {
        x &= 0x1fffffUll;
        x = (x | x << 32Ull) & 0x1f00000000ffffUll;
        x = (x | x << 16Ull) & 0x1f0000ff0000ffUll;
        x = (x | x << 8Ull) & 0x100f00f00f00f00fUll;
        x = (x | x << 4Ull) & 0x10c30c30c30c30c3Ull;
        x = (x | x << 2Ull) & 0x1249249249249249Ull;
        return x;
    }

    // 10bit number and 2 interleaving bits (for 32 bits)
    template<>
    inline u32 expand_bits<u32, 2>(u32 x) {
        x &= 0x3ffU;
        x = (x | x << 16U) & 0x30000ffU;
        x = (x | x << 8U) & 0x300f00fU;
        x = (x | x << 4U) & 0x30c30c3U;
        x = (x | x << 2U) & 0x9249249U;
        return x;
    }

    // 16 bit number 1 interleaving bit (for 32bits)
    template<>
    inline u32 expand_bits<u32, 1>(u32 x) {
        x = (x | (x << 8U)) & 0x00FF00FFU;
        x = (x | (x << 4U)) & 0x0F0F0F0FU;
        x = (x | (x << 2U)) & 0x33333333U;
        x = (x | (x << 1U)) & 0x55555555U;
        return x;
    }

    // 32 bit number 1 interleaving bit (for 64bits)
    template<>
    inline u64 expand_bits<u64, 1>(u64 x) {
        x = (x | (x << 16Ull)) & 0x0000FFFF0000FFFFull;
        x = (x | (x << 8Ull)) & 0x00FF00FF00FF00FFull;
        x = (x | (x << 4Ull)) & 0x0F0F0F0F0F0F0F0Full;
        x = (x | (x << 2Ull)) & 0x3333333333333333ull;
        x = (x | (x << 1Ull)) & 0x5555555555555555ull;
        return x;
    }

    template<>
    inline u32 expand_bits<u32, 0>(u32 x) {
        return x;
    }

    template<>
    inline u64 expand_bits<u64, 0>(u64 x) {
        return x;
    }

    template<class inttype, int interleaving>
    inttype contract_bits(inttype);

    template<>
    inline u64 contract_bits<u64, 2>(u64 src) {
        // src = src & 0x9249249249249249;
        src = (src | (src >> 2Ull)) & 0x30c30c30c30c30c3Ull;
        src = (src | (src >> 4Ull)) & 0xf00f00f00f00f00fUll;
        src = (src | (src >> 8Ull)) & 0x00ff0000ff0000ffUll;
        src = (src | (src >> 16Ull)) & 0xffff00000000ffffUll;
        src = (src | (src >> 32Ull)) & 0x00000000ffffffffUll;
        return src;
    }

    template<>
    inline u64 contract_bits<u64, 1>(u64 src) {
        src = (src | (src >> 1Ull)) & 0x3333333333333333Ull;
        src = (src | (src >> 2Ull)) & 0x0f0f0f0f0f0f0f0fUll;
        src = (src | (src >> 4Ull)) & 0x00ff00ff00ff00ffUll;
        src = (src | (src >> 8Ull)) & 0x0000ffff0000ffffUll;
        src = (src | (src >> 16Ull)) & 0x00000000ffffffffUll;
        return src;
    }

    template<>
    inline u64 contract_bits<u64, 0>(u64 src) {
        return src;
    }

    template<>
    inline u32 contract_bits<u32, 2>(u32 src) {
        src = (src | src >> 2U) & 0x30C30C3U;
        src = (src | src >> 4U) & 0xF00F00FU;
        src = (src | src >> 8U) & 0xFF0000FFU;
        src = (src | src >> 16U) & 0xFFFFU;
        return src;
    }

    template<>
    inline u32 contract_bits<u32, 1>(u32 src) {
        src = (src | (src >> 1U)) & 0x33333333U;
        src = (src | (src >> 2U)) & 0x0f0f0f0fU;
        src = (src | (src >> 4U)) & 0x00ff00ffU;
        src = (src | (src >> 8U)) & 0x0000ffffU;
        return src;
    }

    template<>
    inline u32 contract_bits<u32, 0>(u32 src) {
        return src;
    }

} // namespace shamrock::sfc::bmi
