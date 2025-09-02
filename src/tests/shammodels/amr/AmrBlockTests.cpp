// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/common/amr/AMRBlock.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammodels/amr/AMRBlock", test_amr_block_coords, 1) {

    using Block = shammodels::amr::AMRBlock<f64_3, i64_3, 1>;

    std::vector<std::array<u32, 3>> test_coords{};

    for (u32 ix = 0; ix < Block::Nside; ix++) {
        for (u32 iy = 0; iy < Block::Nside; iy++) {
            for (u32 iz = 0; iz < Block::Nside; iz++) {

                std::array<u32, 3> res = Block::get_coord(Block::get_index({ix, iy, iz}));
                REQUIRE_EQUAL(res[0], ix);
                REQUIRE_EQUAL(res[1], iy);
                REQUIRE_EQUAL(res[2], iz);
            }
        }
    }

    for (u32 i = 0; i < Block::block_size; i++) {
        u32 j = Block::get_index(Block::get_coord(i));
        REQUIRE_EQUAL(i, j);
    }
}
