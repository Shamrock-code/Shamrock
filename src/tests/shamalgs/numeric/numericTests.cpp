// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "numericTests.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>

TestStart(Unittest, "shamalgs/numeric/stream_compact", streamcompactalg, 1) {
    TestStreamCompact test((TestStreamCompact::vFunctionCall) shamalgs::numeric::stream_compact);
    test.check();
}

TestStart(Unittest, "shamalgs/numeric/stream_compact(usm)", streamcompactalgusm, 1) {
    TestStreamCompactUSM test(
        (TestStreamCompactUSM::vFunctionCall) shamalgs::numeric::stream_compact);
    test.check();
}

TestStart(Unittest, "shamalgs/numeric/device_histogram", devicehistogram, 1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    // Case 1: Normal histogram
    {
        std::vector<double> values = {0.5, 1.5, 2.5, 3.5, 2.1, 1.9, 0.1, 3.9};
        std::vector<u64> expected  = {2, 2, 2, 2};

        sham::DeviceBuffer<double> d_values(values.size(), dev_sched);
        d_values.copy_from_stdvec(values);

        sham::DeviceBuffer<u64> d_counts = shamalgs::numeric::device_histogram(
            dev_sched, d_bin_edges, nbins, d_values, values.size());

        REQUIRE_EQUAL(d_counts.copy_to_stdvec(), expected);
    }

    // Case 2: Empty values list
    {
        std::vector<double> empty_values = {};
        std::vector<u64> expected_empty{0, 0, 0, 0};

        sham::DeviceBuffer<double> d_empty_values(empty_values.size(), dev_sched);
        d_empty_values.copy_from_stdvec(empty_values);

        sham::DeviceBuffer<u64> d_counts_empty
            = shamalgs::numeric::device_histogram(dev_sched, d_bin_edges, nbins, d_empty_values, 0);

        REQUIRE_EQUAL(d_counts_empty.copy_to_stdvec(), expected_empty);
    }

    // Case 3: Values outside bin range
    {
        std::vector<double> out_values = {-1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0};
        // Only 0.5, 1.5, 2.5, 3.5 are in range, so expect 1 in each bin
        std::vector<u64> expected_out = {1, 1, 1, 1};

        sham::DeviceBuffer<double> d_out_values(out_values.size(), dev_sched);
        d_out_values.copy_from_stdvec(out_values);

        sham::DeviceBuffer<u64> d_counts_out = shamalgs::numeric::device_histogram(
            dev_sched, d_bin_edges, nbins, d_out_values, out_values.size());

        REQUIRE_EQUAL(d_counts_out.copy_to_stdvec(), expected_out);
    }
}
