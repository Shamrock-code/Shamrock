// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/sfc/morton.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/sfc/morton/min-max", mortonminmaxval, 1) {

    using Morton32 = shamrock::sfc::MortonCodes<u32, 3>;
    using Morton64 = shamrock::sfc::MortonCodes<u64, 3>;

    f32 zerof32 = 0;
    f32 onef32  = 1;

    u32 m_0_32   = Morton32::coord_to_morton(zerof32, zerof32, zerof32);
    u32 m_max_32 = Morton32::coord_to_morton(onef32, onef32, onef32);

    u64 m_0_64   = Morton64::coord_to_morton(zerof32, zerof32, zerof32);
    u64 m_max_64 = Morton64::coord_to_morton(onef32, onef32, onef32);

    REQUIRE_EQUAL_NAMED("min morton 64 == b0", m_0_64, 0x0);
    REQUIRE_EQUAL_NAMED("max morton 64 == b63x1", m_max_64, 0x7fffffffffffffff);
    REQUIRE_EQUAL_NAMED("min morton 32 == b0x0", m_0_32, 0x0);
    REQUIRE_EQUAL_NAMED("max morton 32 == b30x1", m_max_32, 0x3fffffff);
}
