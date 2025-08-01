// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shammodels/sph/modules/KillParticles.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

TestStart(Unittest, "shambackends/KillParticles:basic", KillParticles_basic, 1) {
    using T = f64;
    using namespace shamrock;
    using namespace shammodels::sph::modules;

    // 1. Create PatchDataLayout
    patch::PatchDataLayout layout;
    layout.add_field<T>("single_var", 1);
    layout.add_field<T>("multi_var", 2);

    std::vector<T> in_1 = {0, 1, 2, 3, 4};
    std::vector<T> in_2 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};

    // 2. Create PatchData with 5 particles: xyz = (i, i, i)
    patch::PatchData pdat(layout);
    pdat.resize(5);
    {
        auto &field = pdat.get_field<T>(layout.get_field_idx<T>("single_var"));
        field.override(in_1, in_1.size());
    }
    {
        auto &field = pdat.get_field<T>(layout.get_field_idx<T>("multi_var"));
        field.override(in_2, in_2.size());
    }
    REQUIRE_EQUAL(pdat.get_obj_cnt(), 5);

    // 3. Create PatchDataLayerRefs referencing this PatchData (patch id 0)
    auto patchdatas = std::make_shared<solvergraph::PatchDataLayerRefs>("patchdatas", "patchdatas");
    patchdatas->patchdatas.add_obj(0, std::ref(pdat));

    // 4. Create DistributedBuffers<u32> with indices {0, 4} (remove first and last)
    auto part_to_remove = std::make_shared<solvergraph::DistributedBuffers<u32>>(
        "part_to_remove", "part_to_remove");
    sham::DeviceBuffer<u32> idx_buf(2, shamsys::instance::get_compute_scheduler_ptr());
    std::vector<u32> idxs = {0, 4};
    idx_buf.copy_from_stdvec(idxs);
    part_to_remove->buffers.add_obj(0, std::move(idx_buf));

    // 5. Set up KillParticles, set edges, call _impl_evaluate_internal()
    KillParticles kill_node;
    kill_node.set_edges(part_to_remove, patchdatas);
    kill_node.evaluate();

    // 6. Assert PatchData now has 3 particles 1,2,3

    std::vector<T> expected_field_1 = {1, 2, 3};
    std::vector<T> expected_field_2 = {1, 1, 2, 2, 3, 3};
    REQUIRE_EQUAL(pdat.get_obj_cnt(), 3);
    {
        auto &field           = pdat.get_field<T>(layout.get_field_idx<T>("single_var"));
        std::vector<T> result = field.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected_field_1);
    }
    {
        auto &field           = pdat.get_field<T>(layout.get_field_idx<T>("multi_var"));
        std::vector<T> result = field.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected_field_2);
    }
}
