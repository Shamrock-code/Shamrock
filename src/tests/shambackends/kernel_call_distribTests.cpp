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
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <utility>
#include <vector>

TestStart(Unittest, "distributed_data_kernel_call_test", testing_func_ddref, 1) {

    using T = f64;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    shambase::DistributedData<sham::DeviceBuffer<T>> rho_field;
    rho_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    rho_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    rho_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    rho_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    shambase::DistributedData<sham::DeviceBuffer<T>> uint_field;
    uint_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    uint_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    uint_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    uint_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    shambase::DistributedData<sham::DeviceBuffer<T>> P_field;
    P_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    P_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    P_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    P_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    shambase::DistributedData<sham::DeviceBuffer<T>> cs_field;
    cs_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    cs_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    cs_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    cs_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    std::vector<T> rho_ref0 = {1, 2, 3, 4};
    std::vector<T> rho_ref1 = {2, 3, 4, 5};
    std::vector<T> rho_ref2 = {6, 7, 8, 9};
    std::vector<T> rho_ref3 = {1, 2, 3, 4};

    rho_field.get(0).copy_from_stdvec(rho_ref0);
    rho_field.get(1).copy_from_stdvec(rho_ref1);
    rho_field.get(2).copy_from_stdvec(rho_ref2);
    rho_field.get(3).copy_from_stdvec(rho_ref3);

    std::vector<T> uint_ref0 = {1, 2, 3, 4};
    std::vector<T> uint_ref1 = {2, 3, 4, 5};
    std::vector<T> uint_ref2 = {6, 7, 8, 9};
    std::vector<T> uint_ref3 = {1, 2, 3, 4};

    uint_field.get(0).copy_from_stdvec(uint_ref0);
    uint_field.get(1).copy_from_stdvec(uint_ref1);
    uint_field.get(2).copy_from_stdvec(uint_ref2);
    uint_field.get(3).copy_from_stdvec(uint_ref3);

    std::vector<T> P_ref0 = rho_ref0;
    std::vector<T> P_ref1 = rho_ref1;
    std::vector<T> P_ref2 = rho_ref2;
    std::vector<T> P_ref3 = rho_ref3;

    std::vector<T> cs_ref0 = uint_ref0;
    std::vector<T> cs_ref1 = uint_ref1;
    std::vector<T> cs_ref2 = uint_ref2;
    std::vector<T> cs_ref3 = uint_ref3;

    shambase::DistributedData<u32> sizes
        = rho_field.map<u32>([](u64, sham::DeviceBuffer<double, sham::device> &buf) {
              return buf.get_size();
          });

    const auto &rho_field_const  = rho_field;
    const auto &uint_field_const = uint_field;

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{rho_field_const, uint_field_const},
        sham::DDMultiRef{P_field, cs_field},
        sizes,
        [](u32 i, const T *rho, const T *U, T *P, T *cs) {
            T r = rho[i];
            T u = U[i];

            P[i]  = r;
            cs[i] = u;
        });

    REQUIRE_EQUAL(P_field.get(0).copy_to_stdvec(), P_ref0);
    REQUIRE_EQUAL(P_field.get(1).copy_to_stdvec(), P_ref1);
    REQUIRE_EQUAL(P_field.get(2).copy_to_stdvec(), P_ref2);
    REQUIRE_EQUAL(P_field.get(3).copy_to_stdvec(), P_ref3);

    REQUIRE_EQUAL(cs_field.get(0).copy_to_stdvec(), cs_ref0);
    REQUIRE_EQUAL(cs_field.get(1).copy_to_stdvec(), cs_ref1);
    REQUIRE_EQUAL(cs_field.get(2).copy_to_stdvec(), cs_ref2);
    REQUIRE_EQUAL(cs_field.get(3).copy_to_stdvec(), cs_ref3);
}

TestStart(Unittest, "distributed_data_kernel_call_hndl_test", testing_func_ddref_hndl, 1) {

    using T = f64;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    shambase::DistributedData<sham::DeviceBuffer<T>> rho_field;
    rho_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    rho_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    rho_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    rho_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    shambase::DistributedData<sham::DeviceBuffer<T>> uint_field;
    uint_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    uint_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    uint_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    uint_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    shambase::DistributedData<sham::DeviceBuffer<T>> P_field;
    P_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    P_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    P_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    P_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    shambase::DistributedData<sham::DeviceBuffer<T>> cs_field;
    cs_field.add_obj(0, sham::DeviceBuffer<T>{4, dev_sched});
    cs_field.add_obj(1, sham::DeviceBuffer<T>{4, dev_sched});
    cs_field.add_obj(2, sham::DeviceBuffer<T>{4, dev_sched});
    cs_field.add_obj(3, sham::DeviceBuffer<T>{4, dev_sched});

    std::vector<T> rho_ref0 = {1, 2, 3, 4};
    std::vector<T> rho_ref1 = {2, 3, 4, 5};
    std::vector<T> rho_ref2 = {6, 7, 8, 9};
    std::vector<T> rho_ref3 = {1, 2, 3, 4};

    rho_field.get(0).copy_from_stdvec(rho_ref0);
    rho_field.get(1).copy_from_stdvec(rho_ref1);
    rho_field.get(2).copy_from_stdvec(rho_ref2);
    rho_field.get(3).copy_from_stdvec(rho_ref3);

    std::vector<T> uint_ref0 = {1, 2, 3, 4};
    std::vector<T> uint_ref1 = {2, 3, 4, 5};
    std::vector<T> uint_ref2 = {6, 7, 8, 9};
    std::vector<T> uint_ref3 = {1, 2, 3, 4};

    uint_field.get(0).copy_from_stdvec(uint_ref0);
    uint_field.get(1).copy_from_stdvec(uint_ref1);
    uint_field.get(2).copy_from_stdvec(uint_ref2);
    uint_field.get(3).copy_from_stdvec(uint_ref3);

    std::vector<T> P_ref0 = rho_ref0;
    std::vector<T> P_ref1 = rho_ref1;
    std::vector<T> P_ref2 = rho_ref2;
    std::vector<T> P_ref3 = rho_ref3;

    std::vector<T> cs_ref0 = uint_ref0;
    std::vector<T> cs_ref1 = uint_ref1;
    std::vector<T> cs_ref2 = uint_ref2;
    std::vector<T> cs_ref3 = uint_ref3;

    shambase::DistributedData<u32> sizes
        = rho_field.map<u32>([](u64, sham::DeviceBuffer<double, sham::device> &buf) {
              return buf.get_size();
          });

    const auto &rho_field_const  = rho_field;
    const auto &uint_field_const = uint_field;

    sham::distributed_data_kernel_call_hndl(
        dev_sched,
        sham::DDMultiRef{rho_field_const, uint_field_const},
        sham::DDMultiRef{P_field, cs_field},
        sizes,
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

    REQUIRE_EQUAL(P_field.get(0).copy_to_stdvec(), P_ref0);
    REQUIRE_EQUAL(P_field.get(1).copy_to_stdvec(), P_ref1);
    REQUIRE_EQUAL(P_field.get(2).copy_to_stdvec(), P_ref2);
    REQUIRE_EQUAL(P_field.get(3).copy_to_stdvec(), P_ref3);

    REQUIRE_EQUAL(cs_field.get(0).copy_to_stdvec(), cs_ref0);
    REQUIRE_EQUAL(cs_field.get(1).copy_to_stdvec(), cs_ref1);
    REQUIRE_EQUAL(cs_field.get(2).copy_to_stdvec(), cs_ref2);
    REQUIRE_EQUAL(cs_field.get(3).copy_to_stdvec(), cs_ref3);
}
