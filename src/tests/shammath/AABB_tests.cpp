// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/typeAliasVec.hpp"
#include "shammath/AABB.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/AABB::contains", test_contains, 1) {

    // Test case 1: Basic containment in 3D
    {
        shammath::AABB<f32_3> outer{{0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f}};
        shammath::AABB<f32_3> inner{{2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f}};

        REQUIRE(outer.contains(inner));
        REQUIRE(!inner.contains(outer));
    }

    // Test case 2: Identical AABBs
    {
        shammath::AABB<f32_3> box1{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        shammath::AABB<f32_3> box2{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};

        REQUIRE(box1.contains(box2));
        REQUIRE(box2.contains(box1));
    }

    // Test case 3: Touching boundaries (should contain)
    {
        shammath::AABB<f32_3> outer{{0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f}};
        shammath::AABB<f32_3> boundary{{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}};

        REQUIRE(outer.contains(boundary));
    }

    // Test case 4: Exceeding boundaries (should not contain)
    {
        shammath::AABB<f32_3> box1{{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}};
        shammath::AABB<f32_3> box2{{-1.0f, 0.0f, 0.0f}, {4.0f, 5.0f, 5.0f}};
        shammath::AABB<f32_3> box3{{0.0f, 0.0f, 0.0f}, {6.0f, 5.0f, 5.0f}};

        REQUIRE(!box1.contains(box2)); // lower bound exceeds
        REQUIRE(!box1.contains(box3)); // upper bound exceeds
    }

    // Test case 5: Partially overlapping (should not contain)
    {
        shammath::AABB<f32_3> box1{{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}};
        shammath::AABB<f32_3> box2{{3.0f, 3.0f, 3.0f}, {8.0f, 8.0f, 8.0f}};

        REQUIRE(!box1.contains(box2));
        REQUIRE(!box2.contains(box1));
    }

    // Test case 6: Degenerate AABB (zero volume)
    {
        shammath::AABB<f32_3> box{{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}};
        shammath::AABB<f32_3> point{{2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}};

        REQUIRE(box.contains(point));
        REQUIRE(!point.contains(box));
    }

    // Test case 7: 2D AABB test
    {
        shammath::AABB<f32_2> outer{{-5.0f, -5.0f}, {5.0f, 5.0f}};
        shammath::AABB<f32_2> inner{{-2.0f, -2.0f}, {2.0f, 2.0f}};
        shammath::AABB<f32_2> outside{{-6.0f, -6.0f}, {6.0f, 6.0f}};

        REQUIRE(outer.contains(inner));
        REQUIRE(!outer.contains(outside));
        REQUIRE(outside.contains(outer));
    }

    // Test case 8: 4D AABB test
    {
        shammath::AABB<f32_4> outer{{0.0f, 0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f, 10.0f}};
        shammath::AABB<f32_4> inner{{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};

        REQUIRE(outer.contains(inner));
        REQUIRE(!inner.contains(outer));
    }

    // Test case 9: Double precision test
    {
        shammath::AABB<f64_3> outer{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
        shammath::AABB<f64_3> inner{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}};

        REQUIRE(outer.contains(inner));
        REQUIRE(!inner.contains(outer));
    }

    // Test case 10: Edge case - one dimension exactly matching bounds
    {
        shammath::AABB<f32_3> box1{{0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f}};
        shammath::AABB<f32_3> box2{{5.0f, 0.0f, 0.0f}, {7.0f, 10.0f, 10.0f}};

        REQUIRE(box1.contains(box2));
        REQUIRE(!box2.contains(box1));
    }
}
