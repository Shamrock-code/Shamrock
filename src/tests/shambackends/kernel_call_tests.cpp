// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/SourceLocation.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <utility>
#include <vector>

TestStart(Unittest, "shambackends/kernel_call", testing_func_kernel_call_base, 1) {

    using T = f64;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<T> rho_field = sham::DeviceBuffer<T>{4, dev_sched};

    sham::DeviceBuffer<T> uint_field = sham::DeviceBuffer<T>{4, dev_sched};

    sham::DeviceBuffer<T> P_field = sham::DeviceBuffer<T>{4, dev_sched};

    sham::DeviceBuffer<T> cs_field = sham::DeviceBuffer<T>{4, dev_sched};

    std::vector<T> rho_ref = {1, 2, 3, 4};
    rho_field.copy_from_stdvec(rho_ref);

    std::vector<T> uint_ref = {1, 2, 3, 4};
    uint_field.copy_from_stdvec(uint_ref);

    std::vector<T> P_ref = rho_ref;
    P_field.copy_from_stdvec(P_ref);

    std::vector<T> cs_ref = uint_ref;
    cs_field.copy_from_stdvec(cs_ref);

    u32 size = rho_field.get_size();

    const auto &rho_field_const  = rho_field;
    const auto &uint_field_const = uint_field;

    sham::kernel_call(
        dev_sched->get_queue(),
        sham::MultiRef{rho_field_const, uint_field_const},
        sham::MultiRef{P_field, cs_field},
        size,
        [](u32 i,
           const T *__restrict rho,
           const T *__restrict U,
           T *__restrict P,
           T *__restrict cs) {
            T r = rho[i];
            T u = U[i];

            P[i]  = r;
            cs[i] = u;
        });

    REQUIRE_EQUAL(P_field.copy_to_stdvec(), P_ref);
    REQUIRE_EQUAL(cs_field.copy_to_stdvec(), cs_ref);
}

TestStart(Unittest, "shambackends/kernel_call_hndl", testing_func_kernel_call_hndl_base, 1) {

    using T = f64;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<T> rho_field = sham::DeviceBuffer<T>{4, dev_sched};

    sham::DeviceBuffer<T> uint_field = sham::DeviceBuffer<T>{4, dev_sched};

    sham::DeviceBuffer<T> P_field = sham::DeviceBuffer<T>{4, dev_sched};

    sham::DeviceBuffer<T> cs_field = sham::DeviceBuffer<T>{4, dev_sched};

    std::vector<T> rho_ref = {1, 2, 3, 4};
    rho_field.copy_from_stdvec(rho_ref);

    std::vector<T> uint_ref = {1, 2, 3, 4};
    uint_field.copy_from_stdvec(uint_ref);

    std::vector<T> P_ref = rho_ref;
    P_field.copy_from_stdvec(P_ref);

    std::vector<T> cs_ref = uint_ref;
    cs_field.copy_from_stdvec(cs_ref);

    u32 size = rho_field.get_size();

    const auto &rho_field_const  = rho_field;
    const auto &uint_field_const = uint_field;

    sham::kernel_call_hndl(
        dev_sched->get_queue(),
        sham::MultiRef{rho_field_const, uint_field_const},
        sham::MultiRef{P_field, cs_field},
        size,
        [](u32 n,
           const T *__restrict rho,
           const T *__restrict U,
           T *__restrict P,
           T *__restrict cs) {
            return [=](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> item) {
                    T r = rho[item.get_linear_id()];
                    T u = U[item.get_linear_id()];

                    P[item.get_linear_id()]  = r;
                    cs[item.get_linear_id()] = u;
                });
            };
        });

    REQUIRE_EQUAL(P_field.copy_to_stdvec(), P_ref);
    REQUIRE_EQUAL(cs_field.copy_to_stdvec(), cs_ref);
}
