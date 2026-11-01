// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/vec.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace {

    /// A scheduler with a 2x2x2 base grid of (initially empty) patches holding a f64_3 field
    std::unique_ptr<PatchScheduler> make_test_scheduler() {
        auto layout = std::make_shared<shamrock::patch::PatchDataLayerLayout>();
        layout->add_field<f64_3>("xyz", 1);

        auto sched = std::make_unique<PatchScheduler>(layout, 1000, 1);
        sched->make_patch_base_grid<3>({2, 2, 2});
        return sched;
    }

    std::vector<u64> sorted(std::vector<u64> v) {
        std::sort(v.begin(), v.end());
        return v;
    }

} // namespace

NEW_TEST(Unittest, "shamrock/scheduler/PatchScheduler::for_each_patch_data_ordered", 1) {
    using namespace shamrock::patch;

    auto sched_ptr        = make_test_scheduler();
    PatchScheduler &sched = *sched_ptr;

    std::vector<u64> unordered_ids;
    sched.for_each_patch_data([&](u64 id_patch, Patch, PatchDataLayer &) {
        unordered_ids.push_back(id_patch);
    });

    std::vector<u64> ordered_ids;
    sched.for_each_patch_data_ordered([&](u64 id_patch, Patch cur_p, PatchDataLayer &) {
        REQUIRE_EQUAL(cur_p.id_patch, id_patch);
        ordered_ids.push_back(id_patch);
    });

    REQUIRE_EQUAL(ordered_ids.size(), size_t(8));
    REQUIRE(std::is_sorted(ordered_ids.begin(), ordered_ids.end()));
    REQUIRE(ordered_ids == sorted(unordered_ids));
}

NEW_TEST(Unittest, "shamrock/scheduler/PatchScheduler::for_each_patchdata_nonempty_ordered", 1) {
    using namespace shamrock::patch;

    auto sched_ptr        = make_test_scheduler();
    PatchScheduler &sched = *sched_ptr;

    std::vector<u64> all_ids;
    sched.for_each_patch_data_ordered([&](u64 id_patch, Patch, PatchDataLayer &) {
        all_ids.push_back(id_patch);
    });

    // only fill some patches, given out of order
    std::vector<u64> filled = {all_ids[6], all_ids[1], all_ids[3]};
    for (u64 id : filled) {
        sched.patch_data.get_pdat(id).resize(1);
    }

    std::vector<u64> visited;
    sched.for_each_patchdata_nonempty_ordered([&](const Patch cur_p, PatchDataLayer &pdat) {
        REQUIRE_EQUAL(pdat.get_obj_cnt(), u32(1));
        visited.push_back(cur_p.id_patch);
    });

    REQUIRE(visited == sorted(filled));
}

NEW_TEST(Unittest, "shamrock/scheduler/PatchScheduler::rankgather_field(id order)", 1) {
    using namespace shamrock::patch;

    auto sched_ptr        = make_test_scheduler();
    PatchScheduler &sched = *sched_ptr;

    // patch i holds (i + 1) objects whose value is its patch id, so that the gathered buffer
    // is in id order only if the patches were concatenated in id order
    std::vector<u64> ids;
    sched.for_each_patch_data_ordered([&](u64 id_patch, Patch, PatchDataLayer &) {
        ids.push_back(id_patch);
    });

    std::vector<f64_3> expected;
    for (u32 i = 0; i < ids.size(); i++) {
        u64 id               = ids[i];
        PatchDataLayer &pdat = sched.patch_data.get_pdat(id);
        pdat.resize(i + 1);
        f64_3 val = {f64(id), f64(id), f64(id)};
        pdat.get_field<f64_3>(0).get_buf().fill(val);
        for (u32 j = 0; j < i + 1; j++) {
            expected.push_back(val);
        }
    }

    std::unique_ptr<sycl::buffer<f64_3>> gathered = sched.rankgather_field<f64_3>(0);
    REQUIRE(bool(gathered));
    REQUIRE_EQUAL(gathered->size(), expected.size());

    sycl::host_accessor acc{*gathered, sycl::read_only};
    bool same = true;
    for (u32 i = 0; i < expected.size(); i++) {
        same = same && (acc[i].x() == expected[i].x());
    }
    REQUIRE(same);
}
