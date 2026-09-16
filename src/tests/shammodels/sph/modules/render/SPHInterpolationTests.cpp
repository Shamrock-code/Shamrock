// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHInterpolationTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Unit tests for SPHInterpolation vs host double-loop reference.
 *
 */

#include "shammodels/sph/modules/render/SPHInterpolation.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/sph/modules/render/SPHRenderTestCommon.hpp"
#include <memory>

NEW_TEST(Unittest, "shammodels/sph/modules/render/SPHInterpolation:vs_direct", -1) {
    using namespace sph_render_test;
    using namespace shammodels::sph::modules;

    auto global = make_global_dataset();
    auto points = make_slice_points();
    auto ref    = reference_slice(global, points);

    auto loc = make_round_robin_fields(global);

    auto gpart_mass           = make_gpart_mass();
    auto tree_reduction_level = make_tree_reduction_level();
    auto interp_points        = make_query_edge<Tvec>("interp_points", points);
    auto interpolated_field   = make_output_edge();

    auto node = std::make_shared<SPHInterpolation<Tvec, Tscal, shammath::M4>>();
    node->set_edges(
        gpart_mass,
        tree_reduction_level,
        loc.part_counts,
        loc.positions,
        loc.h_part,
        loc.field_data,
        interp_points,
        interpolated_field);
    node->evaluate();

    auto out = interpolated_field->value.copy_to_stdvec();
    REQUIRE_EQUAL(out.size(), ref.size());
    REQUIRE_EQUAL_CUSTOM_COMP(out, ref, [](const auto &a, const auto &b) {
        return almost_equal_vec(a, b);
    });
}
