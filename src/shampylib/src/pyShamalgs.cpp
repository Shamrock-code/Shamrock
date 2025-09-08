// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamalgs.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shambase/time.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shamalgs/random.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/complex.h>

Register_pymod(shamalgslibinit) {

    py::module shamalgs_module = m.def_submodule("algs", "algorithmic library");

    py::class_<std::mt19937>(shamalgs_module, "rng");

    py::class_<shamalgs::impl_param>(shamalgs_module, "impl_param")
        .def(py::init([]() {
            return shamalgs::impl_param{"", ""};
        }))
        .def_readwrite(
            "impl_name",
            &shamalgs::impl_param::impl_name,
            py::return_value_policy::reference_internal)
        .def_readwrite(
            "params", &shamalgs::impl_param::params, py::return_value_policy::reference_internal)
        .def(
            "__str__",
            [](const shamalgs::impl_param &impl_param) {
                return shambase::format(
                    "impl_param(impl_name=\"{}\", params=\"{}\")",
                    impl_param.impl_name,
                    impl_param.params);
            })
        .def("__repr__", [](const shamalgs::impl_param &impl_param) {
            return shambase::format(
                "impl_param(impl_name=\"{}\", params=\"{}\")",
                impl_param.impl_name,
                impl_param.params);
        });

    shamalgs_module.def("gen_seed", [](u64 seed) {
        return std::mt19937(seed);
    });

    shamalgs_module.def("mock_gaussian", [](std::mt19937 &eng) {
        return shamalgs::random::mock_gaussian<f64>(eng);
    });
    shamalgs_module.def("mock_gaussian_f64_2", [](std::mt19937 &eng) {
        return shamalgs::random::mock_gaussian_multidim<f64_2>(eng);
    });
    shamalgs_module.def("mock_gaussian_f64_3", [](std::mt19937 &eng) {
        return shamalgs::random::mock_gaussian_multidim<f64_3>(eng);
    });
    shamalgs_module.def("mock_unit_vector_f64_3", [](std::mt19937 &eng) {
        return shamalgs::random::mock_unit_vector<f64_3>(eng);
    });

    shamalgs_module.def("mock_buffer_f64", [](u64 seed, u32 len, f64 min_bound, f64 max_bound) {
        return shamalgs::random::mock_buffer_usm<f64>(
            shamsys::instance::get_compute_scheduler_ptr(), seed, len, min_bound, max_bound);
    });
    shamalgs_module.def("mock_buffer_u8", [](u64 seed, u32 len, u8 min_bound, u8 max_bound) {
        return shamalgs::random::mock_buffer_usm<u8>(
            shamsys::instance::get_compute_scheduler_ptr(), seed, len, min_bound, max_bound);
    });
    shamalgs_module.def(
        "mock_buffer_f64_2", [](u64 seed, u32 len, f64_2 min_bound, f64_2 max_bound) {
            return shamalgs::random::mock_buffer_usm<f64_2>(
                shamsys::instance::get_compute_scheduler_ptr(), seed, len, min_bound, max_bound);
        });
    shamalgs_module.def(
        "mock_buffer_f64_3", [](u64 seed, u32 len, f64_3 min_bound, f64_3 max_bound) {
            return shamalgs::random::mock_buffer_usm<f64_3>(
                shamsys::instance::get_compute_scheduler_ptr(), seed, len, min_bound, max_bound);
        });

    { // is_all_true

        shamalgs_module.def("is_all_true", [](sham::DeviceBuffer<u8> &buf, u32 len) {
            return shamalgs::primitives::is_all_true(buf, len);
        });

        shamalgs_module.def("benchmark_is_all_true", [](sham::DeviceBuffer<u8> &buf, u32 len) {
            buf.synchronize();
            shambase::Timer timer;
            timer.start();
            bool result = shamalgs::primitives::is_all_true(buf, len);
            buf.synchronize();
            timer.end();
            return timer.elasped_sec();
        });

        shamalgs_module.def(
            "set_impl_is_all_true", [](const std::string &impl, const std::string &param = "") {
                shamalgs::primitives::impl::set_impl_is_all_true(impl, param);
            });

        shamalgs_module.def("get_current_impl_is_all_true", []() {
            return shamalgs::primitives::impl::get_current_impl_is_all_true();
        });

        shamalgs_module.def("get_default_impl_list_is_all_true", []() {
            return shamalgs::primitives::impl::get_default_impl_list_is_all_true();
        });
    }

    { // reductions
        shamalgs_module.def("sum", [](sham::DeviceBuffer<f64> &buf, u32 start_id, u32 end_id) {
            return shamalgs::primitives::sum(
                shamsys::instance::get_compute_scheduler_ptr(), buf, start_id, end_id);
        });

        shamalgs_module.def("benchmark_reduction_sum", [](sham::DeviceBuffer<f64> &buf, u32 len) {
            buf.synchronize();
            shambase::Timer timer;
            timer.start();
            f64 result = shamalgs::primitives::sum(
                shamsys::instance::get_compute_scheduler_ptr(), buf, 0, len);
            timer.end();
            return timer.elasped_sec();
        });

        shamalgs_module.def("benchmark_reduction_sum", [](sham::DeviceBuffer<f32> &buf, u32 len) {
            buf.synchronize();
            shambase::Timer timer;
            timer.start();
            f32 result = shamalgs::primitives::sum(
                shamsys::instance::get_compute_scheduler_ptr(), buf, 0, len);
            timer.end();
            return timer.elasped_sec();
        });

        shamalgs_module.def(
            "set_impl_reduction", [](const std::string &impl, const std::string &param = "") {
                shamalgs::primitives::impl::set_impl_reduction(impl, param);
            });

        shamalgs_module.def("get_current_impl_reduction", []() {
            return shamalgs::primitives::impl::get_current_impl_reduction();
        });

        shamalgs_module.def("get_default_impl_list_reduction", []() {
            return shamalgs::primitives::impl::get_default_impl_list_reduction();
        });
    }

    { // scan_exclusive_sum_in_place

        shamalgs_module.def(
            "scan_exclusive_sum_in_place", [](sham::DeviceBuffer<u32> &buf, u32 len) {
                shamalgs::primitives::scan_exclusive_sum_in_place(buf, len);
            });

        shamalgs_module.def(
            "benchmark_scan_exclusive_sum_in_place", [](sham::DeviceBuffer<u32> &buf, u32 len) {
                buf.synchronize();
                shambase::Timer timer;
                timer.start();
                shamalgs::primitives::scan_exclusive_sum_in_place(buf, len);
                buf.synchronize();
                timer.end();
                return timer.elasped_sec();
            });

        shamalgs_module.def(
            "set_impl_scan_exclusive_sum_in_place",
            [](const std::string &impl, const std::string &param = "") {
                shamalgs::primitives::impl::set_impl_scan_exclusive_sum_in_place(impl, param);
            });

        shamalgs_module.def("get_current_impl_scan_exclusive_sum_in_place", []() {
            return shamalgs::primitives::impl::get_current_impl_scan_exclusive_sum_in_place();
        });

        shamalgs_module.def("get_default_impl_list_scan_exclusive_sum_in_place", []() {
            return shamalgs::primitives::impl::get_default_impl_list_scan_exclusive_sum_in_place();
        });
    }
}
