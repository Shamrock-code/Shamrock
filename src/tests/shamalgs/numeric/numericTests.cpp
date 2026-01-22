// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "numericTests.hpp"
#include "shamalgs/numeric.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

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

        sham::DeviceBuffer<u64> d_counts = shamalgs::numeric::device_histogram_u64(
            dev_sched, d_bin_edges, nbins, d_values, static_cast<u32>(values.size()));

        REQUIRE_EQUAL(d_counts.copy_to_stdvec(), expected);
    }

    // Case 2: Empty values list
    {
        std::vector<double> empty_values = {};
        std::vector<u64> expected_empty{0, 0, 0, 0};

        sham::DeviceBuffer<double> d_empty_values(0, dev_sched);
        d_empty_values.copy_from_stdvec(empty_values);

        sham::DeviceBuffer<u64> d_counts_empty = shamalgs::numeric::device_histogram_u64(
            dev_sched, d_bin_edges, nbins, d_empty_values, 0);

        REQUIRE_EQUAL(d_counts_empty.copy_to_stdvec(), expected_empty);
    }

    // Case 3: Values outside bin range
    {
        std::vector<double> out_values = {-1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0};
        // Only 0.5, 1.5, 2.5, 3.5 are in range, so expect 1 in each bin
        std::vector<u64> expected_out = {1, 1, 1, 1};

        sham::DeviceBuffer<double> d_out_values(out_values.size(), dev_sched);
        d_out_values.copy_from_stdvec(out_values);

        sham::DeviceBuffer<u64> d_counts_out = shamalgs::numeric::device_histogram_u64(
            dev_sched, d_bin_edges, nbins, d_out_values, static_cast<u32>(out_values.size()));

        REQUIRE_EQUAL(d_counts_out.copy_to_stdvec(), expected_out);
    }
}

TestStart(Unittest, "shamalgs/numeric/binned_sum", binnedsum, 1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    // Case 1: Normal binned sum
    {
        std::vector<double> keys   = {0.5, 1.5, 2.5, 3.5, 2.1, 1.9, 0.1, 3.9};
        std::vector<double> values = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
        // Bin 0: keys 0.5, 0.1 -> values 10.0, 70.0 = 80.0
        // Bin 1: keys 1.5, 1.9 -> values 20.0, 60.0 = 80.0
        // Bin 2: keys 2.5, 2.1 -> values 30.0, 50.0 = 80.0
        // Bin 3: keys 3.5, 3.9 -> values 40.0, 80.0 = 120.0
        std::vector<double> expected = {80.0, 80.0, 80.0, 120.0};

        sham::DeviceBuffer<double> d_keys(keys.size(), dev_sched);
        d_keys.copy_from_stdvec(keys);
        sham::DeviceBuffer<double> d_values(values.size(), dev_sched);
        d_values.copy_from_stdvec(values);

        sham::DeviceBuffer<double> d_sums = shamalgs::numeric::binned_sum(
            dev_sched, d_bin_edges, nbins, d_values, d_keys, static_cast<u32>(keys.size()));

        REQUIRE_EQUAL(d_sums.copy_to_stdvec(), expected);
    }

    // Case 2: Empty values list
    {
        std::vector<double> empty_keys   = {};
        std::vector<double> empty_values = {};
        std::vector<double> expected_empty{0.0, 0.0, 0.0, 0.0};

        sham::DeviceBuffer<double> d_empty_keys(0, dev_sched);
        sham::DeviceBuffer<double> d_empty_values(0, dev_sched);

        d_empty_keys.copy_from_stdvec(empty_keys);
        d_empty_values.copy_from_stdvec(empty_values);

        sham::DeviceBuffer<double> d_sums_empty = shamalgs::numeric::binned_sum(
            dev_sched, d_bin_edges, nbins, d_empty_values, d_empty_keys, 0);

        REQUIRE_EQUAL(d_sums_empty.copy_to_stdvec(), expected_empty);
    }

    // Case 3: Keys outside bin range
    {
        std::vector<double> out_keys   = {-1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0};
        std::vector<double> out_values = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};
        // Only keys 0.5, 1.5, 2.5, 3.5 are in range, so expect:
        // Bin 0: 0.5 -> 20.0
        // Bin 1: 1.5 -> 30.0
        // Bin 2: 2.5 -> 40.0
        // Bin 3: 3.5 -> 50.0
        std::vector<double> expected_out = {20.0, 30.0, 40.0, 50.0};

        sham::DeviceBuffer<double> d_out_keys(out_keys.size(), dev_sched);
        d_out_keys.copy_from_stdvec(out_keys);
        sham::DeviceBuffer<double> d_out_values(out_values.size(), dev_sched);
        d_out_values.copy_from_stdvec(out_values);

        sham::DeviceBuffer<double> d_sums_out = shamalgs::numeric::binned_sum(
            dev_sched,
            d_bin_edges,
            nbins,
            d_out_values,
            d_out_keys,
            static_cast<u32>(out_keys.size()));

        REQUIRE_EQUAL(d_sums_out.copy_to_stdvec(), expected_out);
    }
}

TestStart(Unittest, "shamalgs/numeric/binned_average", binnedaverage, 1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    // Case 1: Normal binned sum
    {
        std::vector<double> keys   = {0.5, 1.5, 2.5, 3.5, 2.1, 1.9, 0.1, 3.9, 0.7};
        std::vector<double> values = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 10.0};
        // Bin 0: keys 0.5, 0.1, 0.7 -> values 10.0, 70.0, 10.0 = 90.0 / 3. = 30.0
        // Bin 1: keys 1.5, 1.9 -> values 20.0, 60.0 = 80.0 / 2. = 40.0
        // Bin 2: keys 2.5, 2.1 -> values 30.0, 50.0 = 80.0 / 2. = 40.0
        // Bin 3: keys 3.5, 3.9 -> values 40.0, 80.0 = 120.0 / 2. = 60.0
        std::vector<double> expected = {30.0, 40.0, 40.0, 60.0};

        sham::DeviceBuffer<double> d_keys(keys.size(), dev_sched);
        d_keys.copy_from_stdvec(keys);
        sham::DeviceBuffer<double> d_values(values.size(), dev_sched);
        d_values.copy_from_stdvec(values);

        sham::DeviceBuffer<double> d_sums = shamalgs::numeric::binned_average(
            dev_sched, d_bin_edges, nbins, d_values, d_keys, static_cast<u32>(keys.size()));

        REQUIRE_EQUAL(d_sums.copy_to_stdvec(), expected);
    }

    // Case 2: Empty values list
    {
        std::vector<double> empty_keys   = {};
        std::vector<double> empty_values = {};
        std::vector<double> expected_empty{0.0, 0.0, 0.0, 0.0};

        sham::DeviceBuffer<double> d_empty_keys(0, dev_sched);
        sham::DeviceBuffer<double> d_empty_values(0, dev_sched);

        d_empty_keys.copy_from_stdvec(empty_keys);
        d_empty_values.copy_from_stdvec(empty_values);

        sham::DeviceBuffer<double> d_sums_empty = shamalgs::numeric::binned_average(
            dev_sched, d_bin_edges, nbins, d_empty_values, d_empty_keys, 0);

        REQUIRE_EQUAL(d_sums_empty.copy_to_stdvec(), expected_empty);
    }

    // Case 3: Keys outside bin range
    {
        std::vector<double> out_keys   = {-1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 3.2};
        std::vector<double> out_values = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 10.0};
        // Only keys 0.5, 1.5, 2.5, 3.5, 3.2 are in range, so expect:
        // Bin 0: 0.5 -> 20.0
        // Bin 1: 1.5 -> 30.0
        // Bin 2: 2.5 -> 40.0
        // Bin 3: 3.5, 3.2 -> 50.0 + 10.0 / 2. = 30.0
        std::vector<double> expected_out = {20.0, 30.0, 40.0, 30.0};

        sham::DeviceBuffer<double> d_out_keys(out_keys.size(), dev_sched);
        d_out_keys.copy_from_stdvec(out_keys);
        sham::DeviceBuffer<double> d_out_values(out_values.size(), dev_sched);
        d_out_values.copy_from_stdvec(out_values);

        sham::DeviceBuffer<double> d_sums_out = shamalgs::numeric::binned_average(
            dev_sched,
            d_bin_edges,
            nbins,
            d_out_values,
            d_out_keys,
            static_cast<u32>(out_keys.size()));

        REQUIRE_EQUAL(d_sums_out.copy_to_stdvec(), expected_out);
    }
}

TestStart(Unittest, "shamalgs/numeric/device_histogram_mpi", devicehistogrammpi, -1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;
    u32 world_size                = static_cast<u32>(shamcomm::world_size());
    u32 world_rank                = static_cast<u32>(shamcomm::world_rank());

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    {
        // Generate source data: world_size x std::vector of values
        std::vector<std::vector<double>> source_vec(world_size);
        source_vec[0] = {0.5, 1.5, 2.5, 3.5}; // 1 in each bin from rank 0
        if (world_size > 1) {
            source_vec[1] = {0.1, 1.9, 2.1}; // 1, 1, 1, 0 from rank 1
        }
        if (world_size > 2) {
            for (u32 i = 2; i < world_size; i++) {
                source_vec[i] = {0.7, 3.2}; // 1, 0, 0, 1 from each additional rank
            }
        }

        // Call the MPI histogram alg with source_vec[world_rank]
        sham::DeviceBuffer<double> d_values(source_vec[world_rank].size(), dev_sched);
        d_values.copy_from_stdvec(source_vec[world_rank]);

        sham::DeviceBuffer<u64> d_counts_mpi = shamalgs::numeric::device_histogram_mpi<u64>(
            dev_sched,
            d_bin_edges,
            nbins,
            d_values,
            static_cast<u32>(source_vec[world_rank].size()));

        // Aggregate all values of the source dataset in the reference dataset
        std::vector<double> reference_values;
        for (const auto &vec : source_vec) {
            reference_values.insert(reference_values.end(), vec.begin(), vec.end());
        }

        // Call the non-MPI variant on the reference dataset
        sham::DeviceBuffer<double> d_ref_values(reference_values.size(), dev_sched);
        d_ref_values.copy_from_stdvec(reference_values);

        sham::DeviceBuffer<u64> d_counts_ref = shamalgs::numeric::device_histogram_u64(
            dev_sched, d_bin_edges, nbins, d_ref_values, static_cast<u32>(reference_values.size()));

        REQUIRE_EQUAL(d_counts_mpi.copy_to_stdvec(), d_counts_ref.copy_to_stdvec());
    }
}

TestStart(Unittest, "shamalgs/numeric/device_histogram_u64_mpi", devicehistogramu64mpi, -1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;
    u32 world_size                = static_cast<u32>(shamcomm::world_size());
    u32 world_rank                = static_cast<u32>(shamcomm::world_rank());

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    {
        // Generate source data: world_size x std::vector of values
        std::vector<std::vector<double>> source_vec(world_size);
        source_vec[0] = {0.3, 1.7, 2.8, 3.1}; // 1 in each bin from rank 0
        if (world_size > 1) {
            source_vec[1] = {0.9, 1.1, 2.4, 3.9}; // 1 in each bin from rank 1
        }
        if (world_size > 2) {
            for (u32 i = 2; i < world_size; i++) {
                source_vec[i] = {0.2, 3.7}; // 1, 0, 0, 1 from each additional rank
            }
        }

        // Call the MPI histogram alg with source_vec[world_rank]
        sham::DeviceBuffer<double> d_values(source_vec[world_rank].size(), dev_sched);
        d_values.copy_from_stdvec(source_vec[world_rank]);

        sham::DeviceBuffer<u64> d_counts_mpi = shamalgs::numeric::device_histogram_u64_mpi(
            dev_sched,
            d_bin_edges,
            nbins,
            d_values,
            static_cast<u32>(source_vec[world_rank].size()));

        // Aggregate all values of the source dataset in the reference dataset
        std::vector<double> reference_values;
        for (const auto &vec : source_vec) {
            reference_values.insert(reference_values.end(), vec.begin(), vec.end());
        }

        // Call the non-MPI variant on the reference dataset
        sham::DeviceBuffer<double> d_ref_values(reference_values.size(), dev_sched);
        d_ref_values.copy_from_stdvec(reference_values);

        sham::DeviceBuffer<u64> d_counts_ref = shamalgs::numeric::device_histogram_u64(
            dev_sched, d_bin_edges, nbins, d_ref_values, static_cast<u32>(reference_values.size()));

        REQUIRE_EQUAL(d_counts_mpi.copy_to_stdvec(), d_counts_ref.copy_to_stdvec());
    }
}

TestStart(Unittest, "shamalgs/numeric/device_histogram_u32_mpi", devicehistogramu32mpi, -1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;
    u32 world_size                = static_cast<u32>(shamcomm::world_size());
    u32 world_rank                = static_cast<u32>(shamcomm::world_rank());

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    {
        // Generate source data: world_size x std::vector of values
        std::vector<std::vector<double>> source_vec(world_size);
        source_vec[0] = {0.6, 1.4, 2.9}; // 1, 1, 1, 0 from rank 0
        if (world_size > 1) {
            source_vec[1] = {0.8, 1.2, 3.4}; // 1, 1, 0, 1 from rank 1
        }
        if (world_size > 2) {
            for (u32 i = 2; i < world_size; i++) {
                source_vec[i] = {2.3, 3.8}; // 0, 0, 1, 1 from each additional rank
            }
        }

        // Call the MPI histogram alg with source_vec[world_rank]
        sham::DeviceBuffer<double> d_values(source_vec[world_rank].size(), dev_sched);
        d_values.copy_from_stdvec(source_vec[world_rank]);

        sham::DeviceBuffer<u32> d_counts_mpi = shamalgs::numeric::device_histogram_u32_mpi(
            dev_sched,
            d_bin_edges,
            nbins,
            d_values,
            static_cast<u32>(source_vec[world_rank].size()));

        // Aggregate all values of the source dataset in the reference dataset
        std::vector<double> reference_values;
        for (const auto &vec : source_vec) {
            reference_values.insert(reference_values.end(), vec.begin(), vec.end());
        }

        // Call the non-MPI variant on the reference dataset
        sham::DeviceBuffer<double> d_ref_values(reference_values.size(), dev_sched);
        d_ref_values.copy_from_stdvec(reference_values);

        sham::DeviceBuffer<u32> d_counts_ref = shamalgs::numeric::device_histogram_u32(
            dev_sched, d_bin_edges, nbins, d_ref_values, static_cast<u32>(reference_values.size()));

        REQUIRE_EQUAL(d_counts_mpi.copy_to_stdvec(), d_counts_ref.copy_to_stdvec());
    }
}

TestStart(Unittest, "shamalgs/numeric/binned_sum_mpi", binnedsummpi, -1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;
    u32 world_size                = static_cast<u32>(shamcomm::world_size());
    u32 world_rank                = static_cast<u32>(shamcomm::world_rank());

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    {
        // Generate source data: world_size x std::vector of keys and values
        std::vector<std::vector<double>> source_keys(world_size);
        std::vector<std::vector<double>> source_values(world_size);

        source_keys[0]   = {0.5, 1.5, 2.5, 3.5};
        source_values[0] = {10.0, 20.0, 30.0, 40.0}; // Sum: 10+20+30+40 per bin

        if (world_size > 1) {
            source_keys[1]   = {0.1, 1.9, 2.1, 3.9};
            source_values[1] = {5.0, 15.0, 25.0, 35.0}; // Sum: 5+15+25+35 per bin
        }

        if (world_size > 2) {
            for (u32 i = 2; i < world_size; i++) {
                source_keys[i]   = {0.7, 3.2};
                source_values[i] = {8.0, 12.0}; // 8 to bin 0, 12 to bin 3
            }
        }

        // Call the MPI binned sum with source_vec[world_rank]
        sham::DeviceBuffer<double> d_keys(source_keys[world_rank].size(), dev_sched);
        d_keys.copy_from_stdvec(source_keys[world_rank]);
        sham::DeviceBuffer<double> d_values(source_values[world_rank].size(), dev_sched);
        d_values.copy_from_stdvec(source_values[world_rank]);

        sham::DeviceBuffer<double> d_sums_mpi = shamalgs::numeric::binned_sum_mpi(
            dev_sched,
            d_bin_edges,
            nbins,
            d_values,
            d_keys,
            static_cast<u32>(source_keys[world_rank].size()));

        // Aggregate all values of the source dataset in the reference dataset
        std::vector<double> reference_keys;
        std::vector<double> reference_values;
        for (u32 i = 0; i < world_size; i++) {
            reference_keys.insert(
                reference_keys.end(), source_keys[i].begin(), source_keys[i].end());
            reference_values.insert(
                reference_values.end(), source_values[i].begin(), source_values[i].end());
        }

        // Call the non-MPI variant on the reference dataset
        sham::DeviceBuffer<double> d_ref_keys(reference_keys.size(), dev_sched);
        d_ref_keys.copy_from_stdvec(reference_keys);
        sham::DeviceBuffer<double> d_ref_values(reference_values.size(), dev_sched);
        d_ref_values.copy_from_stdvec(reference_values);

        sham::DeviceBuffer<double> d_sums_ref = shamalgs::numeric::binned_sum(
            dev_sched,
            d_bin_edges,
            nbins,
            d_ref_values,
            d_ref_keys,
            static_cast<u32>(reference_keys.size()));

        REQUIRE_EQUAL(d_sums_mpi.copy_to_stdvec(), d_sums_ref.copy_to_stdvec());
    }
}

TestStart(Unittest, "shamalgs/numeric/binned_average_mpi", binnedaveragempi, -1) {
    std::vector<double> bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0}; // 4 bins: [0,1), [1,2), [2,3), [3,4)
    u64 nbins                     = bin_edges.size() - 1;
    u32 world_size                = static_cast<u32>(shamcomm::world_size());
    u32 world_rank                = static_cast<u32>(shamcomm::world_rank());

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<double> d_bin_edges(bin_edges.size(), dev_sched);
    d_bin_edges.copy_from_stdvec(bin_edges);

    {
        // Generate source data: world_size x std::vector of keys and values
        std::vector<std::vector<double>> source_keys(world_size);
        std::vector<std::vector<double>> source_values(world_size);

        source_keys[0] = {0.5, 1.5, 2.5, 3.5, 0.3};
        source_values[0]
            = {10.0, 20.0, 30.0, 40.0, 30.0}; // Two values in bin 0: 10+30, one in others

        if (world_size > 1) {
            source_keys[1]   = {0.1, 1.9, 2.1, 3.9};
            source_values[1] = {20.0, 40.0, 50.0, 60.0}; // One value per bin
        }

        if (world_size > 2) {
            for (u32 i = 2; i < world_size; i++) {
                source_keys[i]   = {0.7, 3.2, 3.8};
                source_values[i] = {15.0, 25.0, 35.0}; // 15 to bin 0, 25+35 to bin 3
            }
        }

        // Call the MPI binned average with source_vec[world_rank]
        sham::DeviceBuffer<double> d_keys(source_keys[world_rank].size(), dev_sched);
        d_keys.copy_from_stdvec(source_keys[world_rank]);
        sham::DeviceBuffer<double> d_values(source_values[world_rank].size(), dev_sched);
        d_values.copy_from_stdvec(source_values[world_rank]);

        sham::DeviceBuffer<double> d_averages_mpi = shamalgs::numeric::binned_average_mpi(
            dev_sched,
            d_bin_edges,
            nbins,
            d_values,
            d_keys,
            static_cast<u32>(source_keys[world_rank].size()));

        // Aggregate all values of the source dataset in the reference dataset
        std::vector<double> reference_keys;
        std::vector<double> reference_values;
        for (u32 i = 0; i < world_size; i++) {
            reference_keys.insert(
                reference_keys.end(), source_keys[i].begin(), source_keys[i].end());
            reference_values.insert(
                reference_values.end(), source_values[i].begin(), source_values[i].end());
        }

        // Call the non-MPI variant on the reference dataset
        sham::DeviceBuffer<double> d_ref_keys(reference_keys.size(), dev_sched);
        d_ref_keys.copy_from_stdvec(reference_keys);
        sham::DeviceBuffer<double> d_ref_values(reference_values.size(), dev_sched);
        d_ref_values.copy_from_stdvec(reference_values);

        sham::DeviceBuffer<double> d_averages_ref = shamalgs::numeric::binned_average(
            dev_sched,
            d_bin_edges,
            nbins,
            d_ref_values,
            d_ref_keys,
            static_cast<u32>(reference_keys.size()));

        REQUIRE_EQUAL(d_averages_mpi.copy_to_stdvec(), d_averages_ref.copy_to_stdvec());
    }
}
