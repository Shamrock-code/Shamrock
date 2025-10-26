// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/SchedulerPatchData.hpp"
#include "shamrock/scheduler/scheduler_patch_list.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <map>
#include <vector>

TestStart(
    Unittest,
    "shamrock/scheduler/SchedulerPatchData::apply_change_list",
    tetsschedpatchdataapplychangelist,
    -1) {

    using namespace shamsys::instance;
    using namespace shamsys;
    using namespace shamrock::patch;
    using namespace shamrock::scheduler;

    const i32 wsize = shamcomm::world_size();
    const i32 wrank = shamcomm::world_rank();

    u64 npatch        = wsize * 5;
    u64 seed          = 0x123;
    u32 max_ob_ppatch = 1e3;

    std::mt19937 eng(seed);

    SchedulerPatchList plist;

    for (u32 i = 0; i < npatch; i++) {
        Patch p;
        p.id_patch      = i;
        p.node_owner_id = shamalgs::primitives::mock_value(eng, 0, wsize - 1);
        plist.global.push_back(p);
    }

    std::shared_ptr<PatchDataLayerLayout> pdl_ptr = std::make_shared<PatchDataLayerLayout>();
    auto &pdl                                     = *pdl_ptr;

    pdl.add_field<f32_3>("f32_3'", 1);
    pdl.add_field<f32>("f32", 1);
    pdl.add_field<f32_2>("f32_2", 1);
    pdl.add_field<f32_3>("f32_3", 1);
    pdl.add_field<f32_3>("f32_3''", 1);
    pdl.add_field<f32_4>("f32_4", 1);
    pdl.add_field<f32_8>("f32_8", 1);
    pdl.add_field<f32_16>("f32_16", 1);
    pdl.add_field<f64>("f64", 1);
    pdl.add_field<f64_2>("f64_2", 1);
    pdl.add_field<f64_3>("f64_3", 1);
    pdl.add_field<f64_4>("f64_4", 2);
    pdl.add_field<f64_8>("f64_8", 1);
    pdl.add_field<f64_16>("f64_16", 1);
    pdl.add_field<u32>("u32", 1);
    pdl.add_field<u64>("u64", 1);

    std::vector<PatchData> ref_pdat;

    for (u32 i = 0; i < npatch; i++) {
        ref_pdat.push_back(
            PatchData::mock_patchdata(
                eng(), shamalgs::primitives::mock_value(eng, 1_u32, max_ob_ppatch), pdl_ptr));
    }

    PatchCoord pcoord({0, 0, 0}, {0, 0, 0});

    SchedulerPatchData spdat(pdl_ptr, pcoord);
    for (u32 i = 0; i < npatch; i++) {
        if (plist.global[i].node_owner_id == wrank) {
            spdat.owned_data.add_obj(u64(i), ref_pdat[i].duplicate());
        }
    }

    std::vector<i32> map_new_owner;

    LoadBalancingChangeList clist;
    for (u32 i = 0; i < npatch; i++) {
        i32 new_owner = shamalgs::primitives::mock_value(eng, 0, wsize - 1);
        if (new_owner != plist.global[i].node_owner_id) {
            LoadBalancingChangeList::ChangeOp op;
            op.patch_id       = i;
            op.patch_idx      = i;
            op.rank_owner_old = plist.global[i].node_owner_id;
            op.rank_owner_new = new_owner;
            op.tag_comm       = i;

            clist.change_ops.push_back(op);
        }
        map_new_owner.push_back(new_owner);
    }

    spdat.apply_change_list(clist, plist);

    // checking if patch in correct rank
    for (u32 i = 0; i < npatch; i++) {
        bool should_have_patch = (wrank == map_new_owner[i]);
        bool has_patch         = spdat.has_patch(i);

        // test correct patch location
        REQUIRE_EQUAL(should_have_patch, has_patch);

        if (has_patch && should_have_patch) {
            REQUIRE(spdat.get_pdat(i) == ref_pdat[i]);
        }
    }
}
