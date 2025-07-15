// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamrockCtx.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <pybind11/numpy.h>
#include <variant>

template<class T>
class VecToNumpy;

template<class T>
class VecToNumpy {
    public:
    static py::array_t<T> convert(std::vector<T> vec) {

        u32 len = vec.size();

        py::array_t<T> ret({len});
        auto r = ret.mutable_unchecked();

        for (u32 i = 0; i < len; i++) {
            r(i) = vec[i];
        }

        return std::move(ret);
    }
};

template<class T>
class VecToNumpy<sycl::vec<T, 2>> {
    public:
    static py::array_t<T> convert(std::vector<sycl::vec<T, 2>> vec) {

        u32 len = vec.size();

        py::array_t<T> ret({len, 2U});
        auto r = ret.mutable_unchecked();

        for (u32 i = 0; i < len; i++) {
            r(i, 0) = vec[i].x();
            r(i, 1) = vec[i].y();
        }
        return std::move(ret);
    }
};

template<class T>
class VecToNumpy<sycl::vec<T, 3>> {
    public:
    static py::array_t<T> convert(std::vector<sycl::vec<T, 3>> vec) {

        u32 len = vec.size();

        py::array_t<T> ret({len, 3U});
        auto r = ret.mutable_unchecked();

        for (u32 i = 0; i < len; i++) {
            r(i, 0) = vec[i].x();
            r(i, 1) = vec[i].y();
            r(i, 2) = vec[i].z();
        }
        return std::move(ret);
    }
};

template<class T>
class VecToNumpy<sycl::vec<T, 4>> {
    public:
    static py::array_t<T> convert(std::vector<sycl::vec<T, 4>> vec) {

        u32 len = vec.size();

        py::array_t<T> ret({len, 4U});
        auto r = ret.mutable_unchecked();

        for (u32 i = 0; i < len; i++) {
            r(i, 0) = vec[i].x();
            r(i, 1) = vec[i].y();
            r(i, 2) = vec[i].z();
            r(i, 3) = vec[i].w();
        }
        return std::move(ret);
    }
};

template<class T>
class VecToNumpy<sycl::vec<T, 8>> {
    public:
    static py::array_t<T> convert(std::vector<sycl::vec<T, 8>> vec) {

        u32 len = vec.size();

        py::array_t<T> ret({len, 8U});
        auto r = ret.mutable_unchecked();

        for (u32 i = 0; i < len; i++) {
            r(i, 0) = vec[i].s0();
            r(i, 1) = vec[i].s1();
            r(i, 2) = vec[i].s2();
            r(i, 3) = vec[i].s3();
            r(i, 4) = vec[i].s4();
            r(i, 5) = vec[i].s5();
            r(i, 6) = vec[i].s6();
            r(i, 7) = vec[i].s7();
        }
        return std::move(ret);
    }
};

template<class T>
class VecToNumpy<sycl::vec<T, 16>> {
    public:
    static py::array_t<T> convert(std::vector<sycl::vec<T, 16>> vec) {

        u32 len = vec.size();

        py::array_t<T> ret({len, 16U});
        auto r = ret.mutable_unchecked();

        for (u32 i = 0; i < len; i++) {
            r(i, 0)  = vec[i].s0();
            r(i, 1)  = vec[i].s1();
            r(i, 2)  = vec[i].s2();
            r(i, 3)  = vec[i].s3();
            r(i, 4)  = vec[i].s4();
            r(i, 5)  = vec[i].s5();
            r(i, 6)  = vec[i].s6();
            r(i, 7)  = vec[i].s7();
            r(i, 8)  = vec[i].s8();
            r(i, 9)  = vec[i].s9();
            r(i, 10) = vec[i].sA();
            r(i, 11) = vec[i].sB();
            r(i, 12) = vec[i].sC();
            r(i, 13) = vec[i].sD();
            r(i, 14) = vec[i].sE();
            r(i, 15) = vec[i].sF();
        }

        return std::move(ret);
    }
};

template<class T>
void append_to_map(
    std::string key,
    std::vector<std::unique_ptr<shamrock::patch::PatchData>> &lst,
    py::dict &dic_out) {

    std::vector<T> vec;

    auto appender = [&](auto &field) {
        if (field.get_name() == key) {

            shamlog_debug_ln("PyShamrockCTX", "appending field", key);

            {
                auto acc = field.get_buf().copy_to_stdvec();
                u32 len  = field.get_val_cnt();

                for (u32 i = 0; i < len; i++) {
                    vec.push_back(acc[i]);
                }
            }
        }
    };

    // auto list_appender = [&](std::vector<PatchDataField<T>> & fields){
    //     for(auto f : fields){
    //         appender(f);
    //     }
    // };

    for (auto &pdat : lst) {

        if (pdat->get_obj_cnt() > 0) {
            pdat->for_each_field<T>([&](auto &field) {
                appender(field);
            });
        }

        // list_appender(pdat->get_field_list<T>());
    }

    if (!vec.empty()) {
        auto arr = VecToNumpy<T>::convert(vec);

        py::print("adding -> ", key);

        dic_out[key.c_str()] = arr;
    }
}

Register_pymod(pyshamrockctxinit) {

    shamlog_debug_ln("[Py]", "registering shamrock.Context");

    py::class_<ShamrockCtx>(m, "Context")
        .def(py::init<>())
        .def("pdata_layout_new", &ShamrockCtx::pdata_layout_new)
        //.def("pdata_layout_do_double_prec_mode", &ShamrockCtx::pdata_layout_do_double_prec_mode)
        //.def("pdata_layout_do_single_prec_mode", &ShamrockCtx::pdata_layout_do_single_prec_mode)
        .def("pdata_layout_add_field", &ShamrockCtx::pdata_layout_add_field_t)
        .def("pdata_layout_print", &ShamrockCtx::pdata_layout_print)
        .def("init_sched", &ShamrockCtx::init_sched)
        .def("close_sched", &ShamrockCtx::close_sched)
        .def("pdata_layout_print", &ShamrockCtx::pdata_layout_print)
        .def("pdata_layout_print", &ShamrockCtx::pdata_layout_print)
        .def("pdata_layout_print", &ShamrockCtx::pdata_layout_print)
        .def("pdata_layout_print", &ShamrockCtx::pdata_layout_print)
        .def("dump_status", &ShamrockCtx::dump_status)
        .def(
            "scheduler_step",
            [](ShamrockCtx &self, bool do_split_merge, bool do_load_balancing) {
                self.scheduler_step(do_split_merge, do_load_balancing);
            },
            py::arg("do_split_merge")    = true,
            py::arg("do_load_balancing") = true)
        .def(
            "set_coord_domain_bound",
            [](ShamrockCtx &ctx, std::array<f64, 3> min_vals, std::array<f64, 3> max_vals) {
                ctx.set_coord_domain_bound(
                    {f64_3{min_vals[0], min_vals[1], min_vals[2]},
                     f64_3{max_vals[0], max_vals[1], max_vals[2]}});
            })
        .def("collect_data", [](ShamrockCtx &ctx) {
            auto data = ctx.allgather_data();

            std::cout << "collected : " << data.size() << " patches" << std::endl;

            py::dict dic_out;

            for (auto fname : ctx.pdl->get_field_names()) {
                append_to_map<f32>(fname, data, dic_out);
                append_to_map<f32_2>(fname, data, dic_out);
                append_to_map<f32_3>(fname, data, dic_out);
                append_to_map<f32_4>(fname, data, dic_out);
                append_to_map<f32_8>(fname, data, dic_out);
                append_to_map<f32_16>(fname, data, dic_out);
                append_to_map<f64>(fname, data, dic_out);
                append_to_map<f64_2>(fname, data, dic_out);
                append_to_map<f64_3>(fname, data, dic_out);
                append_to_map<f64_4>(fname, data, dic_out);
                append_to_map<f64_8>(fname, data, dic_out);
                append_to_map<f64_16>(fname, data, dic_out);
                append_to_map<u32>(fname, data, dic_out);
                append_to_map<u64>(fname, data, dic_out);
                append_to_map<u32_3>(fname, data, dic_out);
                append_to_map<u64_3>(fname, data, dic_out);
                append_to_map<i64_3>(fname, data, dic_out);
            }

            return dic_out;
        });
}
