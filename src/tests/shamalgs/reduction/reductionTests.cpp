// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/details/reduction/fallbackReduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction.hpp"
#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/sycl2020reduction.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/reduction.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include <map>
#include <random>

using namespace shamalgs::random;

template<class T, class Fct>
void unit_test_reduc_sum(std::string name, Fct &&red_fct) {

    constexpr u32 size_test = 1e4;

    using Prop = shambase::VectorProperties<T>;
    T min_b = Prop::get_min(), max_b = Prop::get_max();

    if constexpr (Prop::is_float_based) {
        max_b = 1e6;
        min_b = -1e6;
    }

    std::vector<T> vals = shamalgs::random::mock_vector<T>(0x1111, size_test, min_b, max_b);

    T sycl_ret, check_val;

    {
        sycl::buffer<T> buf{vals.data(), vals.size()};

        sycl_ret = red_fct(shamsys::instance::get_compute_queue(), buf, 0, size_test);
    }

    {
        check_val = shambase::VectorProperties<T>::get_zero();
        for (auto &f : vals) {
            check_val += f;
        }
    }

    T delt   = (sycl_ret - check_val) / 1e8;
    auto dot = sham::dot(delt, delt);

    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, dot, 0, 1e-9, sham::abs);
}

template<class T, class Fct>
void unit_test_reduc_min(std::string name, Fct &&red_fct) {

    constexpr u32 size_test = 1e4;

    using Prop = shambase::VectorProperties<T>;
    T min_b = Prop::get_min(), max_b = Prop::get_max();

    if constexpr (Prop::is_float_based) {
        max_b = 1e6;
        min_b = -1e6;
    }

    std::vector<T> vals = shamalgs::random::mock_vector<T>(0x1111, size_test, min_b, max_b);

    T sycl_ret, check_val;

    {
        sycl::buffer<T> buf{vals.data(), vals.size()};

        sycl_ret = red_fct(shamsys::instance::get_compute_queue(), buf, 0, size_test);
    }

    {
        check_val = shambase::VectorProperties<T>::get_max();
        for (auto &f : vals) {
            check_val = sham::min(f, check_val);
        }
    }

    T delt   = (sycl_ret - check_val) / 1e8;
    auto dot = sham::dot(delt, delt);

    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, dot, 0, 1e-9, sham::abs);
}

template<class T, class Fct>
void unit_test_reduc_max(std::string name, Fct &&red_fct) {

    constexpr u32 size_test = 1e4;

    using Prop = shambase::VectorProperties<T>;
    T min_b = Prop::get_min(), max_b = Prop::get_max();

    if constexpr (Prop::is_float_based) {
        max_b = 1e6;
        min_b = -1e6;
    }

    std::vector<T> vals = shamalgs::random::mock_vector<T>(0x1111, size_test, min_b, max_b);

    T sycl_ret, check_val;

    {
        sycl::buffer<T> buf{vals.data(), vals.size()};

        sycl_ret = red_fct(shamsys::instance::get_compute_queue(), buf, 0, size_test);
    }

    {
        check_val = shambase::VectorProperties<T>::get_min();
        for (auto &f : vals) {
            check_val = sham::max(f, check_val);
        }
    }

    T delt   = (sycl_ret - check_val) / 1e8;
    auto dot = sham::dot(delt, delt);

    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, dot, 0, 1e-9, sham::abs);
}
template<class T, class Fct>
void unit_test_reduc_sum_usm(std::string name, Fct &&red_fct) {

    constexpr u32 size_test = 1e4;

    using Prop = shambase::VectorProperties<T>;
    T min_b = Prop::get_min(), max_b = Prop::get_max();

    if constexpr (Prop::is_float_based) {
        max_b = 1e6;
        min_b = -1e6;
    }

    std::vector<T> vals = shamalgs::random::mock_vector<T>(0x1111, size_test, min_b, max_b);

    T sycl_ret, check_val;

    {
        sham::DeviceBuffer<T> buf(vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        buf.copy_from_stdvec(vals);
        auto sched = shamsys::instance::get_compute_scheduler_ptr();
        sycl_ret   = red_fct(sched, buf, 0, size_test);
    }

    {
        check_val = shambase::VectorProperties<T>::get_zero();
        for (auto &f : vals) {
            check_val += f;
        }
    }

    T delt   = (sycl_ret - check_val) / 1e8;
    auto dot = sham::dot(delt, delt);

    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, dot, 0, 1e-9, sham::abs);
}
template<class T, class Fct>
void unit_test_reduc_max_usm(std::string name, Fct &&red_fct) {

    constexpr u32 size_test = 1e4;

    using Prop = shambase::VectorProperties<T>;
    T min_b = Prop::get_min(), max_b = Prop::get_max();

    if constexpr (Prop::is_float_based) {
        max_b = 1e6;
        min_b = -1e6;
    }

    std::vector<T> vals = shamalgs::random::mock_vector<T>(0x1111, size_test, min_b, max_b);

    T sycl_ret, check_val;

    {
        sham::DeviceBuffer<T> buf(vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        buf.copy_from_stdvec(vals);
        auto sched = shamsys::instance::get_compute_scheduler_ptr();
        sycl_ret   = red_fct(sched, buf, 0, size_test);
    }

    {
        check_val = shambase::VectorProperties<T>::get_min();
        for (auto &f : vals) {
            check_val = sham::max(f, check_val);
        }
    }

    T delt   = (sycl_ret - check_val) / 1e8;
    auto dot = sham::dot(delt, delt);

    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, dot, 0, 1e-9, sham::abs);
}

template<class T, class Fct>
void unit_test_reduc_min_usm(std::string name, Fct &&red_fct) {

    constexpr u32 size_test = 1e4;

    using Prop = shambase::VectorProperties<T>;
    T min_b = Prop::get_min(), max_b = Prop::get_max();

    if constexpr (Prop::is_float_based) {
        max_b = 1e6;
        min_b = -1e6;
    }

    std::vector<T> vals = shamalgs::random::mock_vector<T>(0x1111, size_test, min_b, max_b);

    T sycl_ret, check_val;

    {
        sham::DeviceBuffer<T> buf(vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        buf.copy_from_stdvec(vals);
        auto sched = shamsys::instance::get_compute_scheduler_ptr();
        sycl_ret   = red_fct(sched, buf, 0, size_test);
    }

    {
        check_val = shambase::VectorProperties<T>::get_max();
        for (auto &f : vals) {
            check_val = sham::min(f, check_val);
        }
    }

    T delt   = (sycl_ret - check_val) / 1e8;
    auto dot = sham::dot(delt, delt);

    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, dot, 0, 1e-9, sham::abs);
}

void unit_test_reduc_sum() {

    unit_test_reduc_sum<f64>(
        "reduction : main (f64)",
        [](sycl::queue &q, sycl::buffer<f64> &buf1, u32 start_id, u32 end_id) -> f64 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        });

    unit_test_reduc_sum<f32>(
        "reduction : main (f32)",
        [](sycl::queue &q, sycl::buffer<f32> &buf1, u32 start_id, u32 end_id) -> f32 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        });

    unit_test_reduc_sum<u32>(
        "reduction : main (u32)",
        [](sycl::queue &q, sycl::buffer<u32> &buf1, u32 start_id, u32 end_id) -> u32 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        });

    unit_test_reduc_sum<f64_3>(
        "reduction : main (f64_3)",
        [](sycl::queue &q, sycl::buffer<f64_3> &buf1, u32 start_id, u32 end_id) -> f64_3 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        });
}

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
void unit_test_reduc_sum_usm_group_impl() {

    unit_test_reduc_sum_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::details::sum_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_sum_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::details::sum_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_sum_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::details::sum_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_sum_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::details::sum_usm_group(sched, buf1, start_id, end_id, 32);
        });
}
void unit_test_reduc_min_usm_group_impl() {

    unit_test_reduc_min_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::details::min_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_min_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::details::min_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_min_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::details::min_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_min_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::details::min_usm_group(sched, buf1, start_id, end_id, 32);
        });
}
void unit_test_reduc_max_usm_group_impl() {

    unit_test_reduc_max_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::details::max_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_max_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::details::max_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_max_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::details::max_usm_group(sched, buf1, start_id, end_id, 32);
        });

    unit_test_reduc_max_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::details::max_usm_group(sched, buf1, start_id, end_id, 32);
        });
}
#endif

void unit_test_reduc_sum_usm_fallback_impl() {

    unit_test_reduc_sum_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::details::sum_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_sum_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::details::sum_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_sum_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::details::sum_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_sum_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::details::sum_usm_fallback(sched, buf1, start_id, end_id);
        });
}
void unit_test_reduc_min_usm_fallback_impl() {

    unit_test_reduc_min_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::details::min_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_min_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::details::min_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_min_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::details::min_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_min_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::details::min_usm_fallback(sched, buf1, start_id, end_id);
        });
}
void unit_test_reduc_max_usm_fallback_impl() {

    unit_test_reduc_max_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::details::max_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_max_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::details::max_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_max_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::details::max_usm_fallback(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_max_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::details::max_usm_fallback(sched, buf1, start_id, end_id);
        });
}
void unit_test_reduc_sum_usm() {

    unit_test_reduc_sum_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::sum(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_sum_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::sum(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_sum_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::sum(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_sum_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::sum(sched, buf1, start_id, end_id);
        });
}
void unit_test_reduc_min_usm() {

    unit_test_reduc_min_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::min(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_min_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::min(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_min_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::min(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_min_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::min(sched, buf1, start_id, end_id);
        });
}
void unit_test_reduc_max_usm() {

    unit_test_reduc_max_usm<f64>(
        "reduction : main (f64)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64> &buf1,
           u32 start_id,
           u32 end_id) -> f64 {
            return shamalgs::reduction::max(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_max_usm<f32>(
        "reduction : main (f32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f32> &buf1,
           u32 start_id,
           u32 end_id) -> f32 {
            return shamalgs::reduction::max(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_max_usm<u32>(
        "reduction : main (u32)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<u32> &buf1,
           u32 start_id,
           u32 end_id) -> u32 {
            return shamalgs::reduction::max(sched, buf1, start_id, end_id);
        });

    unit_test_reduc_max_usm<f64_3>(
        "reduction : main (f64_3)",
        [](const sham::DeviceScheduler_ptr &sched,
           sham::DeviceBuffer<f64_3> &buf1,
           u32 start_id,
           u32 end_id) -> f64_3 {
            return shamalgs::reduction::max(sched, buf1, start_id, end_id);
        });
}

TestStart(Unittest, "shamalgs/reduction/sum", reduc_kernel_utestsum, 1) { unit_test_reduc_sum(); }

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
TestStart(
    Unittest, "shamalgs/reduction/sum(usm:group_impl)", reduc_kernel_utestsum_usm_group_impl, 1) {
    unit_test_reduc_sum_usm_group_impl();
}
TestStart(
    Unittest, "shamalgs/reduction/min(usm:group_impl)", reduc_kernel_utestmin_usm_group_impl, 1) {
    unit_test_reduc_min_usm_group_impl();
}
TestStart(
    Unittest, "shamalgs/reduction/max(usm:group_impl)", reduc_kernel_utestmax_usm_group_impl, 1) {
    unit_test_reduc_max_usm_group_impl();
}
#endif

TestStart(
    Unittest,
    "shamalgs/reduction/sum(usm:fallback_impl)",
    reduc_kernel_utestsum_usm_fallback_impl,
    1) {
    unit_test_reduc_sum_usm_fallback_impl();
}
TestStart(
    Unittest,
    "shamalgs/reduction/min(usm:fallback_impl)",
    reduc_kernel_utestmin_usm_fallback_impl,
    1) {
    unit_test_reduc_min_usm_fallback_impl();
}
TestStart(
    Unittest,
    "shamalgs/reduction/max(usm:fallback_impl)",
    reduc_kernel_utestmax_usm_fallback_impl,
    1) {
    unit_test_reduc_max_usm_fallback_impl();
}
TestStart(Unittest, "shamalgs/reduction/sum(usm)", reduc_kernel_utestsum_usm, 1) {
    unit_test_reduc_sum_usm();
}
TestStart(Unittest, "shamalgs/reduction/min(usm)", reduc_kernel_utestmin_usm, 1) {
    unit_test_reduc_min_usm();
}
TestStart(Unittest, "shamalgs/reduction/max(usm)", reduc_kernel_utestmax_usm, 1) {
    unit_test_reduc_max_usm();
}

void unit_test_reduc_min() {

    unit_test_reduc_min<f64>(
        "reduction : main (f64)",
        [](sycl::queue &q, sycl::buffer<f64> &buf1, u32 start_id, u32 end_id) -> f64 {
            return shamalgs::reduction::min(q, buf1, start_id, end_id);
        });

    unit_test_reduc_min<f32>(
        "reduction : main (f32)",
        [](sycl::queue &q, sycl::buffer<f32> &buf1, u32 start_id, u32 end_id) -> f32 {
            return shamalgs::reduction::min(q, buf1, start_id, end_id);
        });

    unit_test_reduc_min<u32>(
        "reduction : main (u32)",
        [](sycl::queue &q, sycl::buffer<u32> &buf1, u32 start_id, u32 end_id) -> u32 {
            return shamalgs::reduction::min(q, buf1, start_id, end_id);
        });

    unit_test_reduc_min<f64_3>(
        "reduction : main (f64_3)",
        [](sycl::queue &q, sycl::buffer<f64_3> &buf1, u32 start_id, u32 end_id) -> f64_3 {
            return shamalgs::reduction::min(q, buf1, start_id, end_id);
        });
}

TestStart(Unittest, "shamalgs/reduction/min", reduc_kernel_utestmin, 1) { unit_test_reduc_min(); }

void unit_test_reduc_max() {

    unit_test_reduc_max<f64>(
        "reduction : main (f64)",
        [](sycl::queue &q, sycl::buffer<f64> &buf1, u32 start_id, u32 end_id) -> f64 {
            return shamalgs::reduction::max(q, buf1, start_id, end_id);
        });

    unit_test_reduc_max<f32>(
        "reduction : main (f32)",
        [](sycl::queue &q, sycl::buffer<f32> &buf1, u32 start_id, u32 end_id) -> f32 {
            return shamalgs::reduction::max(q, buf1, start_id, end_id);
        });

    unit_test_reduc_max<u32>(
        "reduction : main (u32)",
        [](sycl::queue &q, sycl::buffer<u32> &buf1, u32 start_id, u32 end_id) -> u32 {
            return shamalgs::reduction::max(q, buf1, start_id, end_id);
        });

    unit_test_reduc_max<f64_3>(
        "reduction : main (f64_3)",
        [](sycl::queue &q, sycl::buffer<f64_3> &buf1, u32 start_id, u32 end_id) -> f64_3 {
            return shamalgs::reduction::max(q, buf1, start_id, end_id);
        });
}

TestStart(Unittest, "shamalgs/reduction/max", reduc_kernel_utestmax, 1) { unit_test_reduc_max(); }

//////////////////////////////////////:
// benchmarks
//////////////////////////////////////:

TestStart(Benchmark, "shamalgs/reduction/sum", benchmark_reductionkernels, 1) {

    std::map<std::string, shambase::BenchmarkResult> results;

    using T = f64;

    f64 exp_test = 1.2;
    f64 max_N    = 1e8;

    results.emplace(
        "fallback",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark fallback sum N =", sz);
                sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

                // do op on GPU to force locality on GPU before test
                shamsys::instance::get_compute_queue()
                    .submit([&](sycl::handler &cgh) {
                        sycl::accessor acc{buf, cgh, sycl::read_write};
                        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
                            acc[id] = acc[id] * acc[id];
                        });
                    })
                    .wait();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::FallbackReduction<T>::sum(
                        shamsys::instance::get_compute_queue(), buf, 0, sz);
                    shamsys::instance::get_compute_queue().wait();
                });
            },
            10,
            max_N,
            exp_test));

#ifdef SYCL2020_FEATURE_REDUCTION

    results.emplace(
        "sycl2020",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark sycl2020 sum N =", sz);

                sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

                // do op on GPU to force locality on GPU before test
                shamsys::instance::get_compute_queue()
                    .submit([&](sycl::handler &cgh) {
                        sycl::accessor acc{buf, cgh, sycl::read_write};
                        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
                            acc[id] = acc[id] * acc[id];
                        });
                    })
                    .wait();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::SYCL2020<T>::sum(
                        shamsys::instance::get_compute_queue(), buf, 0, sz);
                    shamsys::instance::get_compute_queue().wait();
                });
            },
            10,
            max_N,
            exp_test));
#endif

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION

    results.emplace(
        "slicegroup8",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark slicegroup8 sum N =", sz);
                sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

                // do op on GPU to force locality on GPU before test
                shamsys::instance::get_compute_queue()
                    .submit([&](sycl::handler &cgh) {
                        sycl::accessor acc{buf, cgh, sycl::read_write};
                        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
                            acc[id] = acc[id] * acc[id];
                        });
                    })
                    .wait();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::GroupReduction<T, 8>::sum(
                        shamsys::instance::get_compute_queue(), buf, 0, sz);
                    shamsys::instance::get_compute_queue().wait();
                });
            },
            10,
            max_N,
            exp_test));

    results.emplace(
        "slicegroup32",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark slicegroup32 sum N =", sz);
                sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

                // do op on GPU to force locality on GPU before test
                shamsys::instance::get_compute_queue()
                    .submit([&](sycl::handler &cgh) {
                        sycl::accessor acc{buf, cgh, sycl::read_write};
                        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
                            acc[id] = acc[id] * acc[id];
                        });
                    })
                    .wait();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::GroupReduction<T, 32>::sum(
                        shamsys::instance::get_compute_queue(), buf, 0, sz);
                    shamsys::instance::get_compute_queue().wait();
                });
            },
            10,
            max_N,
            exp_test));

    results.emplace(
        "slicegroup128",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark slicegroup128 sum N =", sz);
                sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

                // do op on GPU to force locality on GPU before test
                shamsys::instance::get_compute_queue()
                    .submit([&](sycl::handler &cgh) {
                        sycl::accessor acc{buf, cgh, sycl::read_write};
                        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
                            acc[id] = acc[id] * acc[id];
                        });
                    })
                    .wait();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::GroupReduction<T, 128>::sum(
                        shamsys::instance::get_compute_queue(), buf, 0, sz);
                    shamsys::instance::get_compute_queue().wait();
                });
            },
            10,
            max_N,
            exp_test));

    results.emplace(
        "usmgroup128",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark usmgroup128 sum N =", sz);
                std::vector<T> buf = shamalgs::random::mock_vector<T>(0x111, sz);

                auto sched = shamsys::instance::get_compute_scheduler_ptr();

                sham::DeviceBuffer<T> buf1{buf.size(), sched};
                buf1.copy_from_stdvec(buf);

                buf1.synchronize();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::sum_usm_group(sched, buf1, 0, sz, 128);
                    buf1.synchronize();
                });
            },
            10,
            max_N,
            exp_test));

    results.emplace(
        "usmgroup32",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark usmgroup32 sum N =", sz);
                std::vector<T> buf = shamalgs::random::mock_vector<T>(0x111, sz);

                auto sched = shamsys::instance::get_compute_scheduler_ptr();

                sham::DeviceBuffer<T> buf1{buf.size(), sched};
                buf1.copy_from_stdvec(buf);

                buf1.synchronize();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::details::sum_usm_group(sched, buf1, 0, sz, 32);
                    buf1.synchronize();
                });
            },
            10,
            max_N,
            exp_test));
#endif

    results.emplace(
        "usm",
        shambase::benchmark_pow_len(
            [&](u32 sz) {
                logger::raw_ln("benchmark usm sum N =", sz);
                std::vector<T> buf = shamalgs::random::mock_vector<T>(0x111, sz);

                auto sched = shamsys::instance::get_compute_scheduler_ptr();

                sham::DeviceBuffer<T> buf1{buf.size(), sched};
                buf1.copy_from_stdvec(buf);

                buf1.synchronize();

                return shambase::timeit([&]() {
                    T sum = shamalgs::reduction::sum(sched, buf1, 0, sz);
                    buf1.synchronize();
                });
            },
            10,
            max_N,
            exp_test));

    PyScriptHandle hdnl{};

    for (auto &[key, res] : results) {
        hdnl.data()["x"]         = res.counts;
        hdnl.data()[key.c_str()] = res.times;
    }

    hdnl.exec(R"(
        import matplotlib.pyplot as plt
        import numpy as np

        X = np.array(x)

        Y = np.array(fallback)
        plt.plot(X,X/Y,label = "fallback")

        Y = np.array(sycl2020)
        plt.plot(X,X/Y,label = "sycl2020")

        Y = np.array(slicegroup8)
        plt.plot(X,X/Y,label = "slicegroup8")

        Y = np.array(slicegroup32)
        plt.plot(X,X/Y,label = "slicegroup32")

        Y = np.array(slicegroup128)
        plt.plot(X,X/Y,label = "slicegroup128")

        Y = np.array(usmgroup128)
        plt.plot(X,X/Y,label = "usmgroup128")

        Y = np.array(usmgroup32)
        plt.plot(X,X/Y,label = "usmgroup32")

        Y = np.array(usm)
        plt.plot(X,X/Y,label = "usm")

        plt.xlabel("s")
        plt.ylabel("t/N")

        plt.xscale('log')
        plt.yscale('log')

        plt.legend()

        plt.savefig("tests/figures/shamalgsreduc.pdf")
    )");
}
