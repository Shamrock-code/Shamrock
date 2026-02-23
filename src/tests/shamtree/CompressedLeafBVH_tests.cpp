// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

using Tmorton = u64;
using Tvec    = f64_3;
using Tscal   = shambase::VecComponent<Tvec>;

TestStart(Unittest, "shamtree/CompressedLeafBVH(is_empty)", test_compressed_leaf_bvh_is_empty, 1) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    auto bvh = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);
    REQUIRE(bvh.is_empty());
}
