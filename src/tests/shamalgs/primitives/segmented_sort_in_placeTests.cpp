// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/alg_primitives.hpp"
#include "shambase/integer.hpp"
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

template<int I, int ArrSize>
struct OddEvenTransposeSortT {
    template<typename K, typename Comp>
    inline static void Sort(K *keys, const bool *segment_boundary, Comp comp) {
#pragma unroll
        for (int i = 1 & I; i < ArrSize - 1; i += 2)
            if (!segment_boundary[i] && comp(keys[i + 1], keys[i])) {
                std::swap(keys[i], keys[i + 1]);
            }
        OddEvenTransposeSortT<I + 1, ArrSize>::Sort(keys, segment_boundary, comp);
    }
};

TestStart(Unittest, "test_segsort", tmptest, 1) {
    auto sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q    = sched->get_queue();

    std::vector<u32> data    = {41, 67, 34, 0, 39, 24, 78, 58, 62, 64, 5, 81, 45, 27, 61, 91};
    std::vector<u32> offsets = {0, 5, 10, 13, 16};

    sham::DeviceBuffer<u32> data_buf(data.size(), sched);
    data_buf.copy_from_stdvec(data);

    sham::DeviceBuffer<u32> offsets_buf(offsets.size(), sched);
    offsets_buf.copy_from_stdvec(offsets);

    auto print_array = [](const auto &arr, std::vector<u32> &offsets, bool with_header) {
        std::string acc = "";
        std::vector<bool> is_offset(arr.size(), false);
        for (u32 i = 0; i < offsets.size(); i++) {
            is_offset[offsets[i]] = true;
        }
        if (with_header) {
            for (u32 i = 0; i < arr.size(); i++) {
                acc += shambase::format("{:4}", i);
            }
            acc += "\n--------------------------------------------------------------------------\n";
        }

        for (u32 i = 0; i < arr.size(); i++) {
            acc += shambase::format("{:4}", static_cast<u64>(arr[i]));
        }
        acc += "\n";
        for (u32 i = 0; i < arr.size(); i++) {
            if (is_offset[i]) {
                acc += shambase::format("{:>4}", "^");
            } else {
                acc += shambase::format("{:4}", " ");
            }
        }
        acc += "\n";
        shambase::println(acc);
    };

    print_array(data, offsets, true);

    sham::DeviceBuffer<u8> head_flags(data.size(), sched);
    head_flags.fill(0);
    sham::kernel_call(
        q,
        sham::MultiRef{offsets_buf},
        sham::MultiRef{head_flags},
        offsets_buf.get_size() - 1,
        [=](u32 gid, const u32 *__restrict__ offsets, u8 *__restrict__ head_flags) {
            u32 seg_start = offsets[gid];
            if (seg_start > 0) {
                head_flags[seg_start - 1] = 1;
            }
        });

    logger::raw_ln("head_flags:");
    print_array(head_flags.copy_to_stdvec(), offsets, true);

    // block sort groups of n elems in each segs
    static constexpr u32 group_size = 3;
    u32 group_count                 = shambase::group_count(data_buf.get_size(), group_size);
    sham::kernel_call(
        q,
        sham::MultiRef{head_flags},
        sham::MultiRef{data_buf},
        group_count,
        [sz = data_buf.get_size()](
            u32 gid, const u8 *__restrict__ head_flags, u32 *__restrict__ data) {
            std::array<u32, group_size> data_array;
            for (u32 i = 0; i < group_size; i++) {
                u32 idx       = gid * group_size + i;
                data_array[i] = (idx < sz) ? data[idx] : 0;
            }
            std::array<u8, group_size> head_flags_array;
            for (u32 i = 0; i < group_size; i++) {
                u32 idx             = gid * group_size + i;
                head_flags_array[i] = (idx < sz) ? head_flags[idx] : 1_u8;
            }

            shambase::odd_even_transpose_sort_segment_flags<u32, group_size>(
                data_array.data(), head_flags_array.data(), [](u32 a, u32 b) {
                    return a < b;
                });

            for (u32 i = 0; i < group_size; i++) {
                u32 idx = gid * group_size + i;
                if (idx < sz) {
                    data[idx] = data_array[i];
                }
            }
        });

    data = data_buf.copy_to_stdvec();

    logger::raw_ln("data after block sort:");
    print_array(data, offsets, true);
}
