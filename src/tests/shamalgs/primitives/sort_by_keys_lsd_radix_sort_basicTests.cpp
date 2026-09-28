// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/device/details/sort_by_keys_lsd_radix_sort_basic.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

    /// Sort the first `len` elements of (key_data, value_data) with the radix sort and compare the
    /// whole buffers with a std::stable_sort reference : the radix sort is stable, so the result
    /// must match exactly, and elements past `len` must be left untouched
    template<class Tkey>
    void check_against_stable_sort(
        const std::vector<Tkey> &key_data, u32 len, const std::string &case_name) {

        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        u32 buf_size = static_cast<u32>(key_data.size());
        std::vector<u32> value_data(buf_size);
        std::iota(value_data.begin(), value_data.end(), 0);

        std::vector<u32> perm(len);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(), [&](u32 a, u32 b) {
            return key_data[a] < key_data[b];
        });

        std::vector<Tkey> expected_key  = key_data;
        std::vector<u32> expected_value = value_data;
        for (u32 i = 0; i < len; i++) {
            expected_key[i]   = key_data[perm[i]];
            expected_value[i] = value_data[perm[i]];
        }

        sham::DeviceBuffer<Tkey> keys(buf_size, sched);
        sham::DeviceBuffer<u32> values(buf_size, sched);
        if (buf_size > 0) {
            keys.copy_from_stdvec(key_data);
            values.copy_from_stdvec(value_data);
        }

        shamalgs::primitives::device::details::sort_by_keys_lsd_radix_sort_basic(
            sched, keys, values, len);

        std::vector<Tkey> result_key  = keys.copy_to_stdvec();
        std::vector<u32> result_value = values.copy_to_stdvec();

        bool ok = (result_key == expected_key) && (result_value == expected_value);
        if (!ok) {
            shamlog_error_ln(
                "tests",
                "radix sort mismatch, case :",
                case_name,
                "len :",
                len,
                "size :",
                buf_size);
        }
        REQUIRE_NAMED("radix sort == std::stable_sort (" + case_name + ")", ok);
    }

    template<class Tkey>
    void run_all_cases(const std::string &tname) {
        constexpr u32 key_bits = sizeof(Tkey) * 8;
        constexpr Tkey key_max = std::numeric_limits<Tkey>::max();

        auto random_full = [](u32 n, u64 seed) {
            return shamalgs::primitives::mock_vector<Tkey>(seed, n);
        };
        auto random_small_range = [](u32 n, u64 seed) {
            // heavy duplicates : stability matters
            return shamalgs::primitives::mock_vector<Tkey>(seed, n, 0, 3);
        };
        auto all_equal = [](u32 n, u64) {
            return std::vector<Tkey>(n, Tkey(42));
        };
        auto descending = [](u32 n, u64) {
            std::vector<Tkey> v(n);
            for (u32 i = 0; i < n; i++) {
                v[i] = Tkey(n - i);
            }
            return v;
        };
        auto top_digit_only = [](u32 n, u64 seed) {
            // only the most significant digit differs : the last pass decides the order
            std::vector<Tkey> v = shamalgs::primitives::mock_vector<Tkey>(seed, n, 0, 255);
            for (auto &k : v) {
                k = Tkey(k << (key_bits - 8));
            }
            return v;
        };
        auto with_max_and_zero = [](u32 n, u64 seed) {
            std::vector<Tkey> v = shamalgs::primitives::mock_vector<Tkey>(seed, n);
            for (u32 i = 0; i < n; i += 3) {
                v[i] = (i % 2 == 0) ? key_max : Tkey(0);
            }
            return v;
        };

        // every small length (single chunk, and the first few chunks on devices with small
        // minimal chunks), with a few key distributions
        for (u32 len = 0; len <= 300; len++) {
            check_against_stable_sort<Tkey>(random_full(len, len), len, tname + " full small");
            check_against_stable_sort<Tkey>(
                random_small_range(len, len), len, tname + " dup small");
        }

        // lengths around the chunking thresholds of both device types (minimal chunk sizes 64 and
        // 4096, maximal chunk counts reached around 64 * 4096 and 4096 * 512 elements), plus a few
        // primes to get a shorter last chunk
        std::vector<u32> lens
            = {63,
               64,
               65,
               127,
               129,
               4095,
               4096,
               4097,
               8191,
               8193,
               100003,
               262143,
               262145,
               (1u << 21) - 1,
               (1u << 21) + 1};

        for (u32 len : lens) {
            u64 seed = len * 7 + 1;
            check_against_stable_sort<Tkey>(random_full(len, seed), len, tname + " full");
            check_against_stable_sort<Tkey>(random_small_range(len, seed), len, tname + " dup");
            check_against_stable_sort<Tkey>(all_equal(len, seed), len, tname + " equal");
            check_against_stable_sort<Tkey>(descending(len, seed), len, tname + " descending");
            check_against_stable_sort<Tkey>(top_digit_only(len, seed), len, tname + " top digit");
            check_against_stable_sort<Tkey>(
                with_max_and_zero(len, seed), len, tname + " max & zero");
        }

        // sort only a prefix of a larger buffer : the tail must be untouched
        for (u32 len : {0u, 1u, 2u, 1000u, 4097u, 100003u}) {
            check_against_stable_sort<Tkey>(
                random_full(len + 777, len + 3), len, tname + " prefix of larger buffer");
        }
    }

} // namespace

NEW_TEST(Unittest, "shamalgs/primitives/device/details/sort_by_keys_lsd_radix_sort_basic", 1) {
    run_all_cases<u32>("u32");
    run_all_cases<u64>("u64");
}
