// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file batcherOddEvenSortTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tests of the Batcher odd-even mergesort, at power of two lengths and beyond
 *
 */

#include "shamalgs/details/algorithm/batcherOddEvenSort.hpp"
#include "shamalgs/primitives/gen_buffer_index.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

namespace {

    /// Lengths to sweep, deliberately dominated by values that are not powers of two
    const std::vector<u32> test_lengths
        = {0,  1,  2,   3,   4,   5,   7,    8,    9,    15,   16,   17,   31,
           33, 63, 100, 255, 256, 257, 1000, 1023, 1024, 1025, 4097, 65537};

    /// Check that `vals` is a permutation of [0, len)
    inline bool is_permutation_of_iota(const std::vector<u32> &vals) {
        std::vector<u32> sorted = vals;
        std::sort(sorted.begin(), sorted.end());
        std::vector<u32> expected(sorted.size());
        std::iota(expected.begin(), expected.end(), 0);
        return sorted == expected;
    }

    /**
     * @brief Run both implementations on the same input and cross-check them
     *
     * The key bound is kept far below the length on purpose, so that the large cases are
     * dominated by duplicate keys. The device result is required to match the host reference
     * exactly rather than merely being sorted: both run the same comparators in the same
     * order, so the (unstable) permutation they produce must be identical.
     */
    template<class Tkey>
    void check_length(u32 len, Tkey key_bound) {

        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        std::vector<Tkey> keys_in
            = shamalgs::primitives::mock_vector<Tkey>(0x111, len, Tkey{0}, key_bound);

        sham::DeviceBuffer<Tkey> buf_key(len, sched);
        buf_key.copy_from_stdvec(keys_in);

        sham::DeviceBuffer<u32> buf_vals = shamalgs::primitives::gen_buffer_index(sched, len);

        shamalgs::algorithm::details::sort_by_key_batcher_odd_even(sched, buf_key, buf_vals, len);

        std::vector<Tkey> sorted_keys = buf_key.copy_to_stdvec();
        std::vector<u32> sorted_vals  = buf_vals.copy_to_stdvec();

        // the host reference, fed the very same input
        std::vector<Tkey> ref_keys = keys_in;
        std::vector<u32> ref_vals(len);
        std::iota(ref_vals.begin(), ref_vals.end(), 0);
        shamalgs::algorithm::details::sort_by_key_batcher_odd_even_host_reference(
            ref_keys, ref_vals);

        bool sort_ok = std::is_sorted(sorted_keys.begin(), sorted_keys.end());

        bool check_map = true;
        for (u32 i = 0; i < len; i++) {
            check_map = check_map && (sorted_keys[i] == keys_in[sorted_vals[i]]);
        }

        REQUIRE_NAMED("is sorted", sort_ok);
        REQUIRE_NAMED("values permutation ok", check_map);
        REQUIRE_NAMED(
            "values are a permutation of the indexes", is_permutation_of_iota(sorted_vals));
        REQUIRE_NAMED("keys match the host reference", sorted_keys == ref_keys);
        REQUIRE_NAMED("values match the host reference", sorted_vals == ref_vals);
    }

    /// Check the host reference on its own, so that a cross-check failure is attributable
    template<class Tkey>
    void check_length_host_reference(u32 len, Tkey key_bound) {

        std::vector<Tkey> keys
            = shamalgs::primitives::mock_vector<Tkey>(0x111, len, Tkey{0}, key_bound);
        std::vector<Tkey> keys_in = keys;

        std::vector<u32> vals(len);
        std::iota(vals.begin(), vals.end(), 0);

        shamalgs::algorithm::details::sort_by_key_batcher_odd_even_host_reference(keys, vals);

        bool check_map = true;
        for (u32 i = 0; i < len; i++) {
            check_map = check_map && (keys[i] == keys_in[vals[i]]);
        }

        REQUIRE_NAMED("is sorted", std::is_sorted(keys.begin(), keys.end()));
        REQUIRE_NAMED("values permutation ok", check_map);
        REQUIRE_NAMED("values are a permutation of the indexes", is_permutation_of_iota(vals));
    }

} // namespace

NEW_TEST(Unittest, "shamalgs/algorithm/details/batcherOddEvenSort_host_reference", 1) {
    for (u32 len : test_lengths) {
        check_length_host_reference<u32>(len, 1U << 12U);
        check_length_host_reference<u64>(len, 1UL << 40UL);
        check_length_host_reference<f64>(len, 1.0);
    }
}

NEW_TEST(Unittest, "shamalgs/algorithm/details/batcherOddEvenSort", 1) {
    for (u32 len : test_lengths) {
        check_length<u32>(len, 1U << 12U);
    }
}

NEW_TEST(Unittest, "shamalgs/algorithm/details/batcherOddEvenSort_u64_keys", 1) {
    for (u32 len : test_lengths) {
        check_length<u64>(len, 1UL << 40UL);
    }
}

NEW_TEST(Unittest, "shamalgs/algorithm/details/batcherOddEvenSort_f64_keys", 1) {
    for (u32 len : test_lengths) {
        check_length<f64>(len, 1.0);
    }
}
