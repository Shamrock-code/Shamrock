// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/alg_primitives.hpp"
#include "shambase/integer.hpp"
#include "shambase/term_colors.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamalgs/primitives/segmented_sort_in_place.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>

TestStart(
    Unittest, "shamalgs/primitives/segmented_sort_in_place", test_segmented_sort_in_place, 1) {

    auto test_run = []() {
        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        { // empty dataset
            sham::DeviceBuffer<u32> buf(0, sched);
            sham::DeviceBuffer<u32> offsets(1, sched);
            offsets.copy_from_stdvec({0});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            REQUIRE_EQUAL(buf.copy_to_stdvec(), std::vector<u32>{});
        }

        { // single segment - already sorted
            std::vector<u32> data = {1, 2, 3, 4, 5};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            sham::DeviceBuffer<u32> offsets(2, sched);
            offsets.copy_from_stdvec({0, 5});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {1, 2, 3, 4, 5};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // single segment - reverse sorted
            std::vector<u32> data = {5, 4, 3, 2, 1};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            sham::DeviceBuffer<u32> offsets(2, sched);
            offsets.copy_from_stdvec({0, 5});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {1, 2, 3, 4, 5};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // single segment - random order
            std::vector<u32> data = {3, 1, 4, 1, 5, 9, 2, 6};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            sham::DeviceBuffer<u32> offsets(2, sched);
            offsets.copy_from_stdvec({0, 8});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {1, 1, 2, 3, 4, 5, 6, 9};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // multiple segments
            std::vector<u32> data = {3, 1, 4, 7, 5, 2, 9, 8, 6};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            // Three segments: [3,1,4], [7,5,2], [9,8,6]
            sham::DeviceBuffer<u32> offsets(4, sched);
            offsets.copy_from_stdvec({0, 3, 6, 9});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {1, 3, 4, 2, 5, 7, 6, 8, 9};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // segments with varying sizes
            std::vector<u32> data = {5, 10, 3, 7, 1, 9, 2, 8, 4, 6};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            // Segments: [5], [10,3,7,1], [9,2,8,4,6]
            sham::DeviceBuffer<u32> offsets(4, sched);
            offsets.copy_from_stdvec({0, 1, 5, 10});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {5, 1, 3, 7, 10, 2, 4, 6, 8, 9};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // segments with empty segments
            std::vector<u32> data = {3, 1, 4, 7, 5, 2};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            // Segments: [], [3,1,4], [], [7,5,2], []
            sham::DeviceBuffer<u32> offsets(6, sched);
            offsets.copy_from_stdvec({0, 0, 3, 3, 6, 6});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {1, 3, 4, 2, 5, 7};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // large dataset with multiple segments
            u32 seg_count  = 100;
            u32 seg_size   = 10000;
            u32 total_size = seg_count * seg_size;
            std::vector<u32> data
                = shamalgs::primitives::mock_vector<u32>(0x123, total_size, 0, 1000000);

            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            std::vector<u32> offsets_vec(seg_count + 1);
            for (u32 i = 0; i <= seg_count; ++i) {
                offsets_vec[i] = i * seg_size;
            }
            sham::DeviceBuffer<u32> offsets(offsets_vec.size(), sched);
            offsets.copy_from_stdvec(offsets_vec);

            // Create reference by sorting each segment manually
            std::vector<u32> expected = data;
            for (u32 i = 0; i < seg_count; ++i) {
                std::sort(expected.begin() + i * seg_size, expected.begin() + (i + 1) * seg_size);
            }

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // single element segments
            std::vector<u32> data = {5, 3, 8, 1, 9};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            // Each element is its own segment
            sham::DeviceBuffer<u32> offsets(6, sched);
            offsets.copy_from_stdvec({0, 1, 2, 3, 4, 5});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            // Each segment has only one element, so no change
            std::vector<u32> expected = {5, 3, 8, 1, 9};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }

        { // all duplicates
            std::vector<u32> data = {5, 5, 5, 5, 5};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            sham::DeviceBuffer<u32> offsets(2, sched);
            offsets.copy_from_stdvec({0, 5});

            shamalgs::primitives::segmented_sort_in_place(buf, offsets);

            std::vector<u32> expected = {5, 5, 5, 5, 5};
            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }
    };

    auto current_impl = shamalgs::primitives::impl::get_current_impl_segmented_sort_in_place();

    for (shamalgs::impl_param impl :
         shamalgs::primitives::impl::get_default_impl_list_segmented_sort_in_place()) {
        shamalgs::primitives::impl::set_impl_segmented_sort_in_place(impl.impl_name, impl.params);
        shamlog_info_ln("tests", "testing implementation:", impl);
        test_run();
    }

    // reset to default
    shamalgs::primitives::impl::set_impl_segmented_sort_in_place(
        current_impl.impl_name, current_impl.params);
}
