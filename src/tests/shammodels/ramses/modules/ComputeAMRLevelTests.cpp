// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shammodels/ramses/modules/ComputeAMRLevel.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

using TgridVec  = i64_3;
using TgridScal = i64;
using TgridUint = u64;

TestStart(Unittest, "shammodels/ramses/modules/ComputeAMRLevel", ComputeAMRLevel, 1) {

    std::vector<TgridUint> levels_test = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TgridUint l0_ref                   = {1 << 20}; // largest block size

    std::vector<TgridVec> block_min_vec
        = shamalgs::primitives::mock_vector<TgridVec>(0x111, levels_test.size());
    std::vector<TgridVec> block_max_vec = {};

    for (size_t i = 0; i < block_min_vec.size(); i++) {
        block_max_vec.push_back(block_min_vec[i] + (l0_ref / (1 << levels_test[i])));
    }

    std::shared_ptr<shamrock::solvergraph::Indexes<u32>> block_counts
        = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");

    block_counts->indexes.add_obj(0, levels_test.size());

    std::shared_ptr<shamrock::solvergraph::ScalarsEdge<TgridVec>> l0_edge
        = std::make_shared<shamrock::solvergraph::ScalarsEdge<TgridVec>>("", "");

    l0_edge->values.add_obj(0, TgridVec{l0_ref, l0_ref, l0_ref});

    std::shared_ptr<shamrock::solvergraph::Field<TgridVec>> block_min
        = std::make_shared<shamrock::solvergraph::Field<TgridVec>>(1, "", "");
    std::shared_ptr<shamrock::solvergraph::Field<TgridVec>> block_max
        = std::make_shared<shamrock::solvergraph::Field<TgridVec>>(1, "", "");
    std::shared_ptr<shamrock::solvergraph::Field<TgridUint>> block_level
        = std::make_shared<shamrock::solvergraph::Field<TgridUint>>(1, "", "");

    block_min->ensure_sizes(block_counts->indexes);
    block_max->ensure_sizes(block_counts->indexes);
    block_level->ensure_sizes(block_counts->indexes);

    block_min->get_buf(0).copy_from_stdvec(block_min_vec);
    block_max->get_buf(0).copy_from_stdvec(block_max_vec);

    shammodels::basegodunov::modules::ComputeAMRLevel<TgridVec> node;
    node.set_edges(block_counts, l0_edge, block_min, block_max, block_level);
    node.evaluate();

    auto recov_levels = block_level->get_buf(0).copy_to_stdvec();

    REQUIRE_EQUAL(recov_levels, levels_test);
}
