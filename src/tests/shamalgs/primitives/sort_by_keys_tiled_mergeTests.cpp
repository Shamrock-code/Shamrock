// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/numeric_limits.hpp"
#include "shamalgs/details/algorithm/tiled_merge_sort.hpp"
#include "shamalgs/details/algorithm/tiled_merge_sort_host.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace {

    /// Accumulated outcome of a sweep, so that a few thousand sorts only produce a handful of
    /// asserts instead of one per case
    struct SweepReport {
        u64 case_count     = 0;
        u64 unsorted_count = 0;
        u64 mismatch_count = 0;
        std::string first_failure;

        void record_failure(const std::string &what) {
            if (first_failure.empty()) {
                first_failure = what;
            }
        }
    };

    /// Describe a case, for the failure message
    std::string case_name(u32 tile_size, u32 len, u32 elems_per_thread) {
        return "tile_size=" + std::to_string(tile_size) + " len=" + std::to_string(len)
               + " elems_per_thread=" + std::to_string(elems_per_thread);
    }

    /// Check that `result` holds the same (key, value) multiset as `expected_zip` and that the
    /// keys are sorted. Ties may be reordered, the sort is not required to be stable.
    void check_result(
        const std::vector<u32> &result_key,
        const std::vector<u32> &result_val,
        std::vector<std::pair<u32, u32>> expected_zip,
        const std::string &name,
        SweepReport &report) {

        u32 len = static_cast<u32>(expected_zip.size());

        if (!std::is_sorted(result_key.begin(), result_key.begin() + len)) {
            report.unsorted_count++;
            report.record_failure("keys not sorted for " + name);
            return;
        }

        std::vector<std::pair<u32, u32>> result_zip(len);
        for (u32 i = 0; i < len; ++i) {
            result_zip[i] = {result_key[i], result_val[i]};
        }

        std::sort(result_zip.begin(), result_zip.end());
        std::sort(expected_zip.begin(), expected_zip.end());

        if (result_zip != expected_zip) {
            report.mismatch_count++;
            report.record_failure("(key, value) multiset differs for " + name);
        }
    }

    /// Build a case, run it through the device tiled merge sort and the host serial reference,
    /// and check both against std::sort
    template<u32 Vt>
    void run_case(
        const sham::DeviceScheduler_ptr &sched,
        u32 len,
        u32 elems_per_thread,
        u32 seed,
        bool with_sentinel_keys,
        SweepReport &device_report,
        SweepReport &host_report) {

        std::vector<u32> key_data = shamalgs::primitives::mock_vector<u32>(seed, len, 0, 1000000);
        std::vector<u32> val_data(len);
        for (u32 i = 0; i < len; ++i) {
            val_data[i] = i;
        }

        if (with_sentinel_keys) {
            // force real keys equal to the padding sentinel used by the tile pass, so that
            // they tie with the padding of a trailing partial tile
            for (u32 i = 0; i < len; i += 3) {
                key_data[i] = shambase::get_max<u32>();
            }
        }

        std::vector<std::pair<u32, u32>> expected_zip(len);
        for (u32 i = 0; i < len; ++i) {
            expected_zip[i] = {key_data[i], val_data[i]};
        }

        std::string name = case_name(Vt, len, elems_per_thread);

        { // device
            sham::DeviceBuffer<u32> keys(std::max(len, 1u), sched);
            sham::DeviceBuffer<u32> values(std::max(len, 1u), sched);
            if (len > 0) {
                keys.copy_from_stdvec(key_data, len);
                values.copy_from_stdvec(val_data, len);
            }

            shamalgs::algorithm::sort_by_key_tiled_merge<u32, u32, Vt>(
                sched, keys, values, len, elems_per_thread);

            device_report.case_count++;
            check_result(
                keys.copy_to_stdvec(), values.copy_to_stdvec(), expected_zip, name, device_report);
        }

        { // host serial reference
            std::vector<u32> host_key = key_data;
            std::vector<u32> host_val = val_data;

            shamalgs::algorithm::tiled_merge_sort_host_serial<u32, u32, Vt>(
                host_key, host_val, elems_per_thread);

            host_report.case_count++;
            check_result(host_key, host_val, expected_zip, name, host_report);
        }
    }

    /// Sweep lengths and chunk sizes for one tile size
    template<u32 Vt>
    void run_sweep(
        const sham::DeviceScheduler_ptr &sched,
        SweepReport &device_report,
        SweepReport &host_report) {

        u32 seed = 0x1000 * Vt;

        for (u32 elems_per_thread : {1u, 7u, 256u}) {

            // dense sweep, covers len < tile_size, tiles that do not divide len, and the
            // lengths that give an odd run count at some merge round
            for (u32 len = 0; len <= 80; ++len) {
                run_case<Vt>(
                    sched, len, elems_per_thread, seed++, false, device_report, host_report);
            }

            // larger, and non power of two, lengths
            for (u32 len : {127u, 256u, 257u, 1000u, 4099u}) {
                run_case<Vt>(
                    sched, len, elems_per_thread, seed++, false, device_report, host_report);
            }

            // padding sentinel ties with real keys
            for (u32 len : {1u, 2u, 37u, 255u, 999u}) {
                run_case<Vt>(
                    sched, len, elems_per_thread, seed++, true, device_report, host_report);
            }
        }
    }

    /// Report a sweep as a small fixed set of asserts
    void report_sweep(const std::string &label, const SweepReport &report) {
        shamlog_info_ln("tests", label, ": ran", report.case_count, "cases");

        if (!report.first_failure.empty()) {
            shamlog_info_ln("tests", label, ": first failure :", report.first_failure);
        }

        REQUIRE_NAMED((label + " ran cases"), report.case_count > 0);
        REQUIRE_EQUAL_NAMED((label + " keys always sorted"), report.unsorted_count, u64(0));
        REQUIRE_EQUAL_NAMED(
            (label + " (key, value) multiset preserved"), report.mismatch_count, u64(0));
    }

} // namespace

NEW_TEST(Unittest, "shamalgs/primitives/sort_by_keys/tiled_merge", 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    SweepReport device_report;
    SweepReport host_report;

    run_sweep<2>(sched, device_report, host_report);
    run_sweep<4>(sched, device_report, host_report);
    run_sweep<8>(sched, device_report, host_report);
    run_sweep<16>(sched, device_report, host_report);

    report_sweep("tiled merge sort (device)", device_report);
    report_sweep("tiled merge sort (host serial)", host_report);
}

NEW_TEST(Unittest, "shamalgs/primitives/sort_by_keys/tiled_merge_errors", 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<u32> keys(8, sched);
    sham::DeviceBuffer<u32> values(8, sched);
    keys.copy_from_stdvec({7, 6, 5, 4, 3, 2, 1, 0});
    values.copy_from_stdvec({0, 1, 2, 3, 4, 5, 6, 7});

    { // elems_per_thread == 0 is rejected
        bool thrown = false;
        try {
            shamalgs::algorithm::sort_by_key_tiled_merge<u32, u32, 4>(sched, keys, values, 8, 0);
        } catch (const std::invalid_argument &) {
            thrown = true;
        }
        REQUIRE_NAMED("elems_per_thread == 0 throws", thrown);
    }

    { // len larger than the buffers is rejected
        bool thrown = false;
        try {
            shamalgs::algorithm::sort_by_key_tiled_merge<u32, u32, 4>(sched, keys, values, 9);
        } catch (const std::invalid_argument &) {
            thrown = true;
        }
        REQUIRE_NAMED("len past the end of the buffers throws", thrown);
    }

    { // sorting only a prefix leaves the tail untouched
        shamalgs::algorithm::sort_by_key_tiled_merge<u32, u32, 4>(sched, keys, values, 4);

        std::vector<u32> expected_key = {4, 5, 6, 7, 3, 2, 1, 0};
        std::vector<u32> expected_val = {3, 2, 1, 0, 4, 5, 6, 7};
        REQUIRE_EQUAL(keys.copy_to_stdvec(), expected_key);
        REQUIRE_EQUAL(values.copy_to_stdvec(), expected_val);
    }
}
