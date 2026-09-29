// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambase/exception.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/string_histogram.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/impl_registry.hpp"
#include "shamalgs/impl_utils.hpp"
#include "shamalgs/primitives/compute_histogram.hpp"
#include "shamalgs/primitives/digit_histogram.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shamalgs/primitives/segmented_sort_in_place.hpp"
#include "shamalgs/primitives/sort_by_key_pow2_len.hpp"
#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shamalgs/random.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/complex.h>
#include <utility>

ON_PYTHON_INIT {
    auto &m = root_module;

    py::module shamalgs_module = m.def_submodule("algs", "algorithmic library");

    py::class_<std::mt19937>(shamalgs_module, "rng");

    py::class_<shamalgs::impl_param>(shamalgs_module, "impl_param")
        .def(py::init([]() {
            return shamalgs::impl_param{.impl_name = "", .params = ""};
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
                return sham::format(
                    "impl_param(impl_name=\"{}\", params=\"{}\")",
                    impl_param.impl_name,
                    impl_param.params);
            })
        .def("__repr__", [](const shamalgs::impl_param &impl_param) {
            return sham::format(
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
    shamalgs_module.def("mock_buffer_u32", [](u64 seed, u32 len, u32 min_bound, u32 max_bound) {
        return shamalgs::random::mock_buffer_usm<u32>(
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

    { // implementation registry
        shamalgs_module.def(
            "get_registered_algs",
            &shamalgs::impl_registry::get_registered_algs,
            R"pbdoc(
    Get the names of every algorithm with a selectable implementation, sorted.

    Each name can be passed as ``alg`` to the other implementation selection functions of this
    module (``get_default_impl_list``, ``get_current_impl``, ``is_impl_set``, ``set_impl`` and
    ``autoselect_impl``).
            )pbdoc");

        shamalgs_module.def(
            "get_default_impl_list",
            &shamalgs::impl_registry::get_default_impl_list,
            py::arg("alg"),
            R"pbdoc(
    Get the list of available implementations of the algorithm ``alg``.

    Each implementation is a JSON string
    ``{"implementation": "<name>", "parameters": {...}}`` that can be passed to ``set_impl``.

    Raises an exception if no algorithm is registered under ``alg``.
            )pbdoc");

        shamalgs_module.def(
            "get_current_impl",
            &shamalgs::impl_registry::get_current_impl,
            py::arg("alg"),
            R"pbdoc(
    Get the current implementation of the algorithm ``alg``.

    The implementation is returned as a JSON string
    ``{"implementation": "<name>", "parameters": {...}}``, or ``"null"`` if none is selected yet.

    Raises an exception if no algorithm is registered under ``alg``.
            )pbdoc");

        shamalgs_module.def(
            "is_impl_set",
            &shamalgs::impl_registry::is_impl_set,
            py::arg("alg"),
            R"pbdoc(
    Whether an implementation of the algorithm ``alg`` is selected.

    An unset algorithm selects its default implementation on first use, or through
    ``autoselect_impl``.

    Raises an exception if no algorithm is registered under ``alg``.
            )pbdoc");

        shamalgs_module.def(
            "set_impl",
            &shamalgs::impl_registry::set_impl,
            py::arg("alg"),
            py::arg("impl"),
            R"pbdoc(
    Select the implementation ``impl`` of the algorithm ``alg``.

    ``impl`` is a JSON string ``{"implementation": "<name>", "parameters": {...}}``, typically
    one of the entries of ``get_default_impl_list(alg)``.

    Raises an exception if no algorithm is registered under ``alg``, or if ``impl`` is not a
    valid implementation of it.
            )pbdoc");

        shamalgs_module.def(
            "autoselect_impl",
            [](const std::string &alg) {
                shamalgs::impl_registry::autoselect_impl(
                    alg, shamsys::instance::get_compute_scheduler_ptr());
            },
            py::arg("alg"),
            R"pbdoc(
    Select the default implementation of the algorithm ``alg`` for the compute device.

    Requires the devices to be initialized (``shamrock.sys.init(...)`` in library mode), and
    raises an exception otherwise, or if no algorithm is registered under ``alg``.
            )pbdoc");
    }

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
            timer.stop();
            return timer.elapsed_sec();
        });

        shamalgs_module.def("set_impl_is_all_true", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_is_all_true(impl);
        });

        shamalgs_module.def("get_current_impl_is_all_true", []() {
            return shamalgs::primitives::impl::get_current_impl_is_all_true();
        });

        shamalgs_module.def("get_default_impl_list_is_all_true", []() {
            return shamalgs::primitives::impl::get_default_impl_list_is_all_true();
        });

        shamalgs_module.def("is_impl_set_is_all_true", []() {
            return shamalgs::primitives::impl::is_impl_set_is_all_true();
        });

        shamalgs_module.def("autoselect_impl_is_all_true", []() {
            shamalgs::primitives::impl::autoselect_impl_is_all_true(
                shamsys::instance::get_compute_scheduler_ptr());
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
            timer.stop();
            return timer.elapsed_sec();
        });

        shamalgs_module.def("benchmark_reduction_sum", [](sham::DeviceBuffer<f32> &buf, u32 len) {
            buf.synchronize();
            shambase::Timer timer;
            timer.start();
            f32 result = shamalgs::primitives::sum(
                shamsys::instance::get_compute_scheduler_ptr(), buf, 0, len);
            timer.stop();
            return timer.elapsed_sec();
        });

        shamalgs_module.def("set_impl_reduction", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_reduction(impl);
        });

        shamalgs_module.def("get_current_impl_reduction", []() {
            return shamalgs::primitives::impl::get_current_impl_reduction();
        });

        shamalgs_module.def("get_default_impl_list_reduction", []() {
            return shamalgs::primitives::impl::get_default_impl_list_reduction();
        });

        shamalgs_module.def("is_impl_set_reduction", []() {
            return shamalgs::primitives::impl::is_impl_set_reduction();
        });

        shamalgs_module.def("autoselect_impl_reduction", []() {
            shamalgs::primitives::impl::autoselect_impl_reduction(
                shamsys::instance::get_compute_scheduler_ptr());
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
                timer.stop();
                return timer.elapsed_sec();
            });

        shamalgs_module.def("set_impl_scan_exclusive_sum_in_place", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_scan_exclusive_sum_in_place(impl);
        });

        shamalgs_module.def("get_current_impl_scan_exclusive_sum_in_place", []() {
            return shamalgs::primitives::impl::get_current_impl_scan_exclusive_sum_in_place();
        });

        shamalgs_module.def("get_default_impl_list_scan_exclusive_sum_in_place", []() {
            return shamalgs::primitives::impl::get_default_impl_list_scan_exclusive_sum_in_place();
        });

        shamalgs_module.def("is_impl_set_scan_exclusive_sum_in_place", []() {
            return shamalgs::primitives::impl::is_impl_set_scan_exclusive_sum_in_place();
        });

        shamalgs_module.def("autoselect_impl_scan_exclusive_sum_in_place", []() {
            shamalgs::primitives::impl::autoselect_impl_scan_exclusive_sum_in_place(
                shamsys::instance::get_compute_scheduler_ptr());
        });
    }

    { // segmented_sort_in_place
        shamalgs_module.def(
            "segmented_sort_in_place",
            [](sham::DeviceBuffer<u32> &buf, const sham::DeviceBuffer<u32> &offsets) {
                shamalgs::primitives::segmented_sort_in_place(buf, offsets);
            });

        shamalgs_module.def(
            "benchmark_segmented_sort_in_place",
            [](sham::DeviceBuffer<u32> &buf, const sham::DeviceBuffer<u32> &offsets) {
                auto buf_copy     = buf.copy();
                auto offsets_copy = offsets.copy();

                buf_copy.synchronize();
                offsets_copy.synchronize();

                shambase::Timer timer;
                timer.start();

                shamalgs::primitives::segmented_sort_in_place(buf_copy, offsets_copy);
                buf_copy.synchronize();
                offsets_copy.synchronize();

                timer.stop();
                return timer.elapsed_sec();
            });

        shamalgs_module.def("set_impl_segmented_sort_in_place", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_segmented_sort_in_place(impl);
        });

        shamalgs_module.def("get_current_impl_segmented_sort_in_place", []() {
            return shamalgs::primitives::impl::get_current_impl_segmented_sort_in_place();
        });

        shamalgs_module.def("get_default_impl_list_segmented_sort_in_place", []() {
            return shamalgs::primitives::impl::get_default_impl_list_segmented_sort_in_place();
        });
    }

    { // sort_by_keys
        shamalgs_module.def(
            "sort_by_keys",
            [](sham::DeviceBuffer<u32> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len) {
                shamalgs::primitives::sort_by_keys(buf_key, buf_values, len);
            });

        shamalgs_module.def(
            "benchmark_sort_by_keys",
            [](sham::DeviceBuffer<u32> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len) {
                auto buf_key_copy    = buf_key.copy();
                auto buf_values_copy = buf_values.copy();

                buf_key_copy.synchronize();
                buf_values_copy.synchronize();

                shambase::Timer timer;
                timer.start();

                shamalgs::primitives::sort_by_keys(buf_key_copy, buf_values_copy, len);
                buf_key_copy.synchronize();
                buf_values_copy.synchronize();

                timer.stop();
                return timer.elapsed_sec();
            });

        shamalgs_module.def("set_impl_sort_by_keys", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_sort_by_keys(impl);
        });

        shamalgs_module.def("get_current_impl_sort_by_keys", []() {
            return shamalgs::primitives::impl::get_current_impl_sort_by_keys();
        });

        shamalgs_module.def("get_default_impl_list_sort_by_keys", []() {
            return shamalgs::primitives::impl::get_default_impl_list_sort_by_keys();
        });

        shamalgs_module.def("is_impl_set_sort_by_keys", []() {
            return shamalgs::primitives::impl::is_impl_set_sort_by_keys();
        });

        shamalgs_module.def("autoselect_impl_sort_by_keys", []() {
            shamalgs::primitives::impl::autoselect_impl_sort_by_keys(
                shamsys::instance::get_compute_scheduler_ptr());
        });
    }

    { // sort_by_key_pow2_len
        shamalgs_module.def(
            "sort_by_key_pow2_len",
            [](sham::DeviceBuffer<u32> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len) {
                shamalgs::primitives::sort_by_key_pow2_len(
                    shamsys::instance::get_compute_scheduler_ptr(), buf_key, buf_values, len);
            });

        shamalgs_module.def(
            "benchmark_sort_by_key_pow2_len",
            [](sham::DeviceBuffer<u32> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len) {
                auto buf_key_copy    = buf_key.copy();
                auto buf_values_copy = buf_values.copy();

                buf_key_copy.synchronize();
                buf_values_copy.synchronize();

                shambase::Timer timer;
                timer.start();

                shamalgs::primitives::sort_by_key_pow2_len(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    buf_key_copy,
                    buf_values_copy,
                    len);
                buf_key_copy.synchronize();
                buf_values_copy.synchronize();

                timer.stop();
                return timer.elapsed_sec();
            });

        shamalgs_module.def("set_impl_sort_by_key_pow2_len", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_sort_by_key_pow2_len(impl);
        });

        shamalgs_module.def("get_current_impl_sort_by_key_pow2_len", []() {
            return shamalgs::primitives::impl::get_current_impl_sort_by_key_pow2_len();
        });

        shamalgs_module.def("get_default_impl_list_sort_by_key_pow2_len", []() {
            return shamalgs::primitives::impl::get_default_impl_list_sort_by_key_pow2_len();
        });

        shamalgs_module.def("is_impl_set_sort_by_key_pow2_len", []() {
            return shamalgs::primitives::impl::is_impl_set_sort_by_key_pow2_len();
        });

        shamalgs_module.def("autoselect_impl_sort_by_key_pow2_len", []() {
            shamalgs::primitives::impl::autoselect_impl_sort_by_key_pow2_len(
                shamsys::instance::get_compute_scheduler_ptr());
        });
    }

    { // digit_histogram
        // dispatch the runtime radix_bits onto the compile time instantiations
        auto digit_histogram_u32 = [](sham::DeviceBuffer<u32> &buf_key,
                                      sham::DeviceBuffer<u32> &buf_hist,
                                      u32 radix_bits,
                                      u32 len) {
            auto sched = shamsys::instance::get_compute_scheduler_ptr();
            switch (radix_bits) {
            case 1:
                shamalgs::primitives::digit_histogram<u32, 1>(sched, buf_key, buf_hist, len);
                break;
            case 2:
                shamalgs::primitives::digit_histogram<u32, 2>(sched, buf_key, buf_hist, len);
                break;
            case 4:
                shamalgs::primitives::digit_histogram<u32, 4>(sched, buf_key, buf_hist, len);
                break;
            case 8:
                shamalgs::primitives::digit_histogram<u32, 8>(sched, buf_key, buf_hist, len);
                break;
            default:
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "radix_bits must be one of 1, 2, 4, 8, got radix_bits = {}", radix_bits));
            }
        };

        shamalgs_module.def(
            "digit_histogram",
            [digit_histogram_u32](sham::DeviceBuffer<u32> &buf_key, u32 radix_bits, u32 len) {
                sham::DeviceBuffer<u32> buf_hist(0, shamsys::instance::get_compute_scheduler_ptr());
                digit_histogram_u32(buf_key, buf_hist, radix_bits, len);
                return buf_hist;
            },
            py::arg("buf_key"),
            py::arg("radix_bits"),
            py::arg("len"),
            R"pbdoc(
    Histograms of every radix digit place of the first ``len`` u32 keys of ``buf_key``.

    Returns a ``DeviceBuffer_u32`` of ``(32 / radix_bits) * 2**radix_bits`` bins, digit place
    major : ``hist[p * 2**radix_bits + digit]`` counts the keys whose digit ``p`` (bits
    ``p * radix_bits`` to ``(p + 1) * radix_bits - 1``) equals ``digit``.
    ``radix_bits`` must be one of 1, 2, 4, 8.
)pbdoc");

        shamalgs_module.def(
            "benchmark_digit_histogram",
            [digit_histogram_u32](sham::DeviceBuffer<u32> &buf_key, u32 radix_bits, u32 len) {
                sham::DeviceBuffer<u32> buf_hist(0, shamsys::instance::get_compute_scheduler_ptr());

                // warmup, which also allocates buf_hist to its final size
                digit_histogram_u32(buf_key, buf_hist, radix_bits, len);
                buf_key.synchronize();
                buf_hist.synchronize();

                shambase::Timer timer;
                timer.start();

                digit_histogram_u32(buf_key, buf_hist, radix_bits, len);
                buf_hist.synchronize();

                timer.stop();
                return timer.elapsed_sec();
            },
            py::arg("buf_key"),
            py::arg("radix_bits"),
            py::arg("len"));
    }

    { // compute_histogram

        shamalgs_module.def("set_impl_compute_histogram", [](const std::string &impl) {
            shamalgs::primitives::impl::set_impl_compute_histogram(impl);
        });

        shamalgs_module.def("get_current_impl_compute_histogram", []() {
            return shamalgs::primitives::impl::get_current_impl_compute_histogram();
        });

        shamalgs_module.def("get_default_impl_list_compute_histogram", []() {
            return shamalgs::primitives::impl::get_default_impl_list_compute_histogram();
        });

        shamalgs_module.def("is_impl_set_compute_histogram", []() {
            return shamalgs::primitives::impl::is_impl_set_compute_histogram();
        });

        shamalgs_module.def("autoselect_impl_compute_histogram", []() {
            shamalgs::primitives::impl::autoselect_impl_compute_histogram(
                shamsys::instance::get_compute_scheduler_ptr());
        });
    }

    shamalgs_module.def(
        "compute_histogram_basic_f64",
        [](sham::DeviceBuffer<f64> &bin_edge_inf,
           sham::DeviceBuffer<f64> &bin_edge_sup,
           sham::DeviceBuffer<f64> &positions) {
            return shamalgs::primitives::compute_histogram_basic<f64>(
                shamsys::instance::get_compute_scheduler_ptr(),
                bin_edge_inf,
                bin_edge_sup,
                positions);
        });
    shamalgs_module.def(
        "compute_histogram_basic_f32",
        [](sham::DeviceBuffer<f32> &bin_edge_inf,
           sham::DeviceBuffer<f32> &bin_edge_sup,
           sham::DeviceBuffer<f32> &positions) {
            return shamalgs::primitives::compute_histogram_basic<f32>(
                shamsys::instance::get_compute_scheduler_ptr(),
                bin_edge_inf,
                bin_edge_sup,
                positions);
        });

    shamalgs_module.def(
        "benchmark_compute_histogram_basic_f64",
        [](sham::DeviceBuffer<f64> &bin_edge_inf,
           sham::DeviceBuffer<f64> &bin_edge_sup,
           sham::DeviceBuffer<f64> &positions) {
            bin_edge_inf.synchronize();
            bin_edge_sup.synchronize();
            positions.synchronize();

            auto run = [&]() {
                auto result = shamalgs::primitives::compute_histogram_basic<f64>(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    bin_edge_inf,
                    bin_edge_sup,
                    positions);
                result.synchronize();
            };

            run();

            return shambase::timeitfor(run);
        });
    shamalgs_module.def(
        "benchmark_compute_histogram_basic_f32",
        [](sham::DeviceBuffer<f32> &bin_edge_inf,
           sham::DeviceBuffer<f32> &bin_edge_sup,
           sham::DeviceBuffer<f32> &positions) {
            bin_edge_inf.synchronize();
            bin_edge_sup.synchronize();
            positions.synchronize();

            auto run = [&]() {
                auto result = shamalgs::primitives::compute_histogram_basic<f32>(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    bin_edge_inf,
                    bin_edge_sup,
                    positions);
                result.synchronize();
            };

            run();

            return shambase::timeitfor(run);
        });

    shamalgs_module.def(
        "string_histogram",
        [](const std::vector<std::string> &inputs, std::string delimiter, bool hash_based) {
            return shamalgs::collective::string_histogram(inputs, std::move(delimiter), hash_based);
        },
        py::arg("inputs"),
        py::arg("delimiter")  = "\n",
        py::arg("hash_based") = false);

    shamalgs_module.def(
        "all_string_histogram",
        [](const std::vector<std::string> &inputs, std::string delimiter, bool hash_based) {
            return shamalgs::collective::all_string_histogram(
                inputs, std::move(delimiter), hash_based);
        },
        py::arg("inputs"),
        py::arg("delimiter")  = "\n",
        py::arg("hash_based") = false);
}
