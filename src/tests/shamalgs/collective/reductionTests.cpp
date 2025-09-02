// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamalgs/random.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <random>

template<class T>
inline void test() {
    u32 wsize              = shamcomm::world_size();
    u32 wrank              = shamcomm::world_rank();
    std::vector<T> vectest = shamalgs::primitives::mock_vector<T>(0x1111, wsize);

    T sum = shamalgs::collective::allreduce_sum(vectest[wrank]);

    T ref_sum{};

    for (T tmp : vectest) {
        ref_sum += tmp;
    }

    REQUIRE_EQUAL_CUSTOM_COMP(sum, ref_sum, sham::equals);
}

TestStart(Unittest, "shamalgs/collective/reduction/allreduce_sum", testsallreducesum, -1) {
    test<f32>();
    test<u32>();
    test<f64_2>();
    test<f64_3>();
}

TestStart(
    Unittest,
    "shamalgs/collective/reduction/reduce_buffer_in_place_sum",
    testsallreducesum_buf,
    -1) {

    u32 size = 50;
    using T  = u32;

    auto source_func = [](u32 i) {
        return i * (shamcomm::world_rank() + 1);
    };
    auto check_func = [](u32 i) {
        return i * (shamcomm::world_size() * (shamcomm::world_size() + 1)) / 2;
    };

    sham::DeviceBuffer<T, sham::host> source(size, shamsys::instance::get_alt_scheduler_ptr());

    {
        sham::EventList depends_list;
        T *ptr = source.get_write_access(depends_list);
        depends_list.wait_and_throw();

        for (u32 i = 0; i < size; i++) {
            ptr[i] = source_func(i);
        }

        source.complete_event_state(sycl::event{});
    }

    sham::DeviceBuffer<T> buf = source.copy_to<sham::device>();

    shamalgs::collective::reduce_buffer_in_place_sum(buf, MPI_COMM_WORLD);

    source.copy_from(buf);

    {
        sham::EventList depends_list;
        const T *ptr = source.get_read_access(depends_list);
        depends_list.wait_and_throw();

        for (u32 i = 0; i < size; i++) {
            REQUIRE_EQUAL(ptr[i], check_func(i));
        }

        source.complete_event_state(sycl::event{});
    }
}
