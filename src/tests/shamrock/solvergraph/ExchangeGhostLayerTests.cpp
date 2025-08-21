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
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/ExchangeGhostLayer.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <random>
#include <vector>

using namespace shamrock::solvergraph;
using namespace shamrock::patch;

TestStart(Unittest, "shamrock/solvergraph/ExchangeGhostLayer", testExchangeGhostLayer, -1) {

    std::string test_name = "shamrock/solvergraph/ExchangeGhostLayer";

    // Create a simple layout with a few fields
    auto layout = std::make_shared<PatchDataLayerLayout>();
    layout->add_field<f32>("density", 1);
    layout->add_field<f32_3>("velocity", 1);
    layout->add_field<u32>("level", 1);

    // Seed for reproducible random data
    u64 seed = 0x1111;
    std::mt19937 eng(seed);

    u32 npatch = 100;

    // Generate a random original rank owner
    std::vector<u32> rank_owner;
    for (u64 i = 0; i < npatch; i++) {
        rank_owner.push_back(
            shamalgs::primitives::mock_value<u32>(eng, 0_u32, u32(shamcomm::world_size() - 1)));
    }

    // create reference data
    shambase::DistributedDataShared<PatchDataLayer> reference_data;
    for (u64 i = 0; i < npatch; i++) {
        // generate a random message
        u32 obj_count   = shamalgs::primitives::mock_value<u32>(eng, 0_u32, 100_u32);
        auto patch_data = PatchDataLayer::mock_patchdata(eng(), obj_count, layout);

        u32 sender   = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);
        u32 receiver = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);

        shamlog_debug_ln(
            test_name,
            "reference_data: ",
            sender,
            " receiver: ",
            receiver,
            " rank sender: ",
            rank_owner[sender],
            " rank receiver: ",
            rank_owner[receiver],
            " pdat: ",
            patch_data.get_obj_cnt());

        reference_data.add_obj(sender, receiver, std::move(patch_data));
    }

    // select the local data
    std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> exchange_gz_edge
        = std::make_shared<shamrock::solvergraph::PatchDataLayerDDShared>("", "");

    reference_data.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
        if (rank_owner[sender] == shamcomm::world_rank()) {
            exchange_gz_edge->patchdatas.add_obj(sender, receiver, pdat.duplicate());

            shamlog_debug_ln(
                test_name,
                "exchange_gz_edge: ",
                sender,
                " receiver: ",
                receiver,
                " rank sender: ",
                rank_owner[sender],
                " rank receiver: ",
                rank_owner[receiver],
                " pdat: ",
                pdat.get_obj_cnt());
        }
    });

    // create the rank owner edge
    auto rank_owner_edge = std::make_shared<ScalarsEdge<u32>>("rank_owner", "rank_owner");
    for (u64 i = 0; i < npatch; i++) {
        rank_owner_edge->values.add_obj(i, u32(rank_owner[i]));
        shamlog_debug_ln(test_name, "rank_owner: ", i, " rank: ", rank_owner[i]);
    }

    std::shared_ptr<ExchangeGhostLayer> exchange_gz_node
        = std::make_shared<ExchangeGhostLayer>(layout);
    exchange_gz_node->set_edges(rank_owner_edge, exchange_gz_edge);

    exchange_gz_node->evaluate();

    shambase::DistributedDataShared<PatchDataLayer> interf_pdat
        = std::move(exchange_gz_edge->patchdatas);

    // generate the check dataset
    shambase::DistributedDataShared<PatchDataLayer> check_dataset{};
    reference_data.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
        if (rank_owner[receiver] == shamcomm::world_rank()) {
            check_dataset.add_obj(sender, receiver, pdat.duplicate());

            shamlog_debug_ln(
                test_name,
                "check_dataset: ",
                sender,
                " receiver: ",
                receiver,
                " rank sender: ",
                rank_owner[sender],
                " rank receiver: ",
                rank_owner[receiver],
                " pdat: ",
                pdat.get_obj_cnt());
        }
    });

    // verify the data
    auto check_dataset_it = check_dataset.begin();
    auto interf_pdat_it   = interf_pdat.begin();
    for (; check_dataset_it != check_dataset.end(); ++check_dataset_it, ++interf_pdat_it) {
        auto [sender, receiver] = check_dataset_it->first;

        auto &pdat        = check_dataset_it->second;
        auto &interf_pdat = interf_pdat_it->second;

        shamlog_debug_ln(
            test_name,
            "sender: ",
            sender,
            " receiver: ",
            receiver,
            "rank sender: ",
            rank_owner[sender],
            " rank receiver: ",
            rank_owner[receiver],
            " pdat: ",
            pdat.get_obj_cnt(),
            " interf_pdat: ",
            interf_pdat.get_obj_cnt());

        REQUIRE(pdat == interf_pdat);
    }
}
