// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/AMRBlock.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammodels/amr/AMRBlock", test_amr_block_coords, 1) {

    using Block = shammodels::amr::AMRBlock<f64_3, i64_3, 1>;

    std::vector<std::array<u32, 3>> test_coords{};

    for (u32 ix = 0; ix < Block::Nside; ix++) {
        for (u32 iy = 0; iy < Block::Nside; iy++) {
            for (u32 iz = 0; iz < Block::Nside; iz++) {

                std::array<u32, 3> res = Block::get_coord(Block::get_index({ix, iy, iz}));
                _AssertEqual(res[0], ix) _AssertEqual(res[1], iy) _AssertEqual(res[2], iz)
            }
        }
    }

    for (u32 i = 0; i < Block::block_size; i++) {
        u32 j = Block::get_index(Block::get_coord(i));
        _AssertEqual(i, j)
    }
}
