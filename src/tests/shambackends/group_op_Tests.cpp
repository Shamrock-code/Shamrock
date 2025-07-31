// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/group_op.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <sycl/sycl.hpp>
#include <vector>

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION

TestStart(Unittest, "sham::sum_over_group", sum_over_group_test, 1) {

    // Initialize data
    std::vector<f64_3> input_data
        = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
    std::vector<f64_3> output_data(input_data.size(), {0.0, 0.0, 0.0});

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    // Create buffers
    sham::DeviceBuffer<f64_3> input_buf(input_data.size(), dev_sched);
    sham::DeviceBuffer<f64_3> output_buf(output_data.size(), dev_sched);

    input_buf.copy_from_stdvec(input_data);
    output_buf.copy_from_stdvec(output_data);

    sham::EventList depends_list;
    auto in  = input_buf.get_read_access(depends_list);
    auto out = output_buf.get_write_access(depends_list);

    // Submit kernel
    auto e = dev_sched->get_queue().submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(input_data.size(), input_data.size()), [=](sycl::nd_item<1> item) {
                sycl::group<1> group = item.get_group();
                f64_3 sum            = sham::sum_over_group(group, in[item.get_global_linear_id()]);
                out[item.get_global_linear_id()] = sum;
            });
    });

    input_buf.complete_event_state(e);
    output_buf.complete_event_state(e);

    // Check results
    output_data = output_buf.copy_to_stdvec();

    f64_3 expected_sum = {0.0, 0.0, 0.0};
    for (auto o : input_data) {
        expected_sum += o;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
        REQUIRE_EQUAL(output_data[i][0], expected_sum[0]);
        REQUIRE_EQUAL(output_data[i][1], expected_sum[1]);
        REQUIRE_EQUAL(output_data[i][2], expected_sum[2]);
    }
}

TestStart(Unittest, "sham::min_over_group", min_over_group_test, 1) {

    // Initialize data
    std::vector<f64_3> input_data
        = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
    std::vector<f64_3> output_data(input_data.size(), {0.0, 0.0, 0.0});

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    // Create buffers
    sham::DeviceBuffer<f64_3> input_buf(input_data.size(), dev_sched);
    sham::DeviceBuffer<f64_3> output_buf(output_data.size(), dev_sched);

    input_buf.copy_from_stdvec(input_data);
    output_buf.copy_from_stdvec(output_data);

    sham::EventList depends_list;
    auto in  = input_buf.get_read_access(depends_list);
    auto out = output_buf.get_write_access(depends_list);

    // Submit kernel
    auto e = dev_sched->get_queue().submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(input_data.size(), input_data.size()), [=](sycl::nd_item<1> item) {
                sycl::group<1> group = item.get_group();
                f64_3 min            = sham::min_over_group(group, in[item.get_global_linear_id()]);
                out[item.get_global_linear_id()] = min;
            });
    });

    input_buf.complete_event_state(e);
    output_buf.complete_event_state(e);

    // Check results
    output_data = output_buf.copy_to_stdvec();

    f64_3 expected_min = input_data[0];
    for (auto o : input_data) {
        expected_min = sham::min(expected_min, o);
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
        REQUIRE_EQUAL(output_data[i][0], expected_min[0]);
        REQUIRE_EQUAL(output_data[i][1], expected_min[1]);
        REQUIRE_EQUAL(output_data[i][2], expected_min[2]);
    }
}

TestStart(Unittest, "sham::max_over_group", max_over_group_test, 1) {

    // Initialize data
    std::vector<f64_3> input_data
        = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
    std::vector<f64_3> output_data(input_data.size(), {0.0, 0.0, 0.0});

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    // Create buffers
    sham::DeviceBuffer<f64_3> input_buf(input_data.size(), dev_sched);
    sham::DeviceBuffer<f64_3> output_buf(output_data.size(), dev_sched);

    input_buf.copy_from_stdvec(input_data);
    output_buf.copy_from_stdvec(output_data);

    sham::EventList depends_list;
    auto in  = input_buf.get_read_access(depends_list);
    auto out = output_buf.get_write_access(depends_list);

    // Submit kernel
    auto e = dev_sched->get_queue().submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(input_data.size(), input_data.size()), [=](sycl::nd_item<1> item) {
                sycl::group<1> group = item.get_group();
                f64_3 max            = sham::max_over_group(group, in[item.get_global_linear_id()]);
                out[item.get_global_linear_id()] = max;
            });
    });

    input_buf.complete_event_state(e);
    output_buf.complete_event_state(e);

    // Check results
    output_data        = output_buf.copy_to_stdvec();
    f64_3 expected_max = input_data[0];
    for (auto o : input_data) {
        expected_max = sham::max(expected_max, o);
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
        REQUIRE_EQUAL(output_data[i][0], expected_max[0]);
        REQUIRE_EQUAL(output_data[i][1], expected_max[1]);
        REQUIRE_EQUAL(output_data[i][2], expected_max[2]);
    }
}

#endif
