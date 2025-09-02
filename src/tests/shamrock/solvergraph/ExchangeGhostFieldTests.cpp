// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/DistributedDataShared.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/ExchangeGhostField.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <random>
#include <vector>

TestStart(Unittest, "shamrock/solvergraph/ExchangeGhostField", testExchangeGhostField, -1) {

    std::string test_name = "shamrock/solvergraph/ExchangeGhostField";

    // Seed for reproducible random data
    u64 seed = 0x2222;
    std::mt19937 eng(seed);

    u32 npatch = 80;

    // Generate a random original rank owner
    std::vector<u32> rank_owner;
    for (u64 i = 0; i < npatch; i++) {
        rank_owner.push_back(
            shamalgs::primitives::mock_value<u32>(eng, 0_u32, u32(shamcomm::world_size() - 1)));
    }

    // create reference data for f32 fields
    shambase::DistributedDataShared<PatchDataField<f32>> reference_data;
    for (u64 i = 0; i < npatch; i++) {
        // generate a random message
        u32 obj_count          = shamalgs::primitives::mock_value<u32>(eng, 1_u32, 100_u32);
        u32 nvar               = shamalgs::primitives::mock_value<u32>(eng, 1_u32, 5_u32);
        std::string field_name = "density";

        auto patch_field = PatchDataField<f32>::mock_field(eng(), obj_count, field_name, nvar);

        u32 sender   = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);
        u32 receiver = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);

        shamlog_debug_ln(
            test_name,
            "reference_data f32: ",
            sender,
            " receiver: ",
            receiver,
            " rank sender: ",
            rank_owner[sender],
            " rank receiver: ",
            rank_owner[receiver],
            " field obj_count: ",
            patch_field.get_obj_cnt(),
            " nvar: ",
            patch_field.get_nvar());

        reference_data.add_obj(sender, receiver, std::move(patch_field));
    }

    // select the local data
    std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<f32>> exchange_field_edge
        = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<f32>>("ghost_field", "GF");

    reference_data.for_each([&](u64 sender, u64 receiver, PatchDataField<f32> &field) {
        if (rank_owner[sender] == shamcomm::world_rank()) {
            exchange_field_edge->patchdata_fields.add_obj(sender, receiver, field.duplicate());

            shamlog_debug_ln(
                test_name,
                "exchange_field_edge f32: ",
                sender,
                " receiver: ",
                receiver,
                " rank sender: ",
                rank_owner[sender],
                " rank receiver: ",
                rank_owner[receiver],
                " field obj_count: ",
                field.get_obj_cnt(),
                " nvar: ",
                field.get_nvar());
        }
    });

    // create the rank owner edge
    auto rank_owner_edge
        = std::make_shared<shamrock::solvergraph::ScalarsEdge<u32>>("rank_owner", "RO");
    for (u64 i = 0; i < npatch; i++) {
        rank_owner_edge->values.add_obj(i, u32(rank_owner[i]));
        shamlog_debug_ln(test_name, "rank_owner f32: ", i, " rank: ", rank_owner[i]);
    }

    std::shared_ptr<shamrock::solvergraph::ExchangeGhostField<f32>> exchange_field_node
        = std::make_shared<shamrock::solvergraph::ExchangeGhostField<f32>>();
    exchange_field_node->set_edges(rank_owner_edge, exchange_field_edge);

    exchange_field_node->evaluate();

    shambase::DistributedDataShared<PatchDataField<f32>> interf_field
        = std::move(exchange_field_edge->patchdata_fields);

    // generate the check dataset
    shambase::DistributedDataShared<PatchDataField<f32>> check_dataset{};
    reference_data.for_each([&](u64 sender, u64 receiver, PatchDataField<f32> &field) {
        if (rank_owner[receiver] == shamcomm::world_rank()) {
            check_dataset.add_obj(sender, receiver, field.duplicate());

            shamlog_debug_ln(
                test_name,
                "check_dataset f32: ",
                sender,
                " receiver: ",
                receiver,
                " rank sender: ",
                rank_owner[sender],
                " rank receiver: ",
                rank_owner[receiver],
                " field obj_count: ",
                field.get_obj_cnt(),
                " nvar: ",
                field.get_nvar());
        }
    });

    // verify the data
    auto check_dataset_it = check_dataset.begin();
    auto interf_field_it  = interf_field.begin();
    for (; check_dataset_it != check_dataset.end(); ++check_dataset_it, ++interf_field_it) {
        auto [sender, receiver] = check_dataset_it->first;

        auto &field            = check_dataset_it->second;
        auto &interf_field_ref = interf_field_it->second;

        shamlog_debug_ln(
            test_name,
            "verification f32 sender: ",
            sender,
            " receiver: ",
            receiver,
            "rank sender: ",
            rank_owner[sender],
            " rank receiver: ",
            rank_owner[receiver],
            " field obj_count: ",
            field.get_obj_cnt(),
            " interf_field obj_count: ",
            interf_field_ref.get_obj_cnt(),
            " field nvar: ",
            field.get_nvar(),
            " interf_field nvar: ",
            interf_field_ref.get_nvar());

        REQUIRE(field.check_field_match(interf_field_ref));
    }
}
