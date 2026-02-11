// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamrockCtx.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/logs/loglevels.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <variant>

template<class T>
inline void register_patch_transform_class(py::module &m, std::string name) {

    shamlog_debug_ln("[Py]", "registering", name);

    using Tclass = shamrock::patch::PatchCoordTransform<T>;

    py::class_<Tclass>(m, name.c_str())
        .def(
            "to_obj_coord",
            [](Tclass &self, shamrock::patch::Patch p) -> shammath::AABB<T> {
                auto tmp = self.to_obj_coord(p);
                return shammath::AABB<T>(tmp.lower, tmp.upper);
            })
        .def("to_obj_coord", [](Tclass &self, shammath::AABB<u64_3> p) -> shammath::AABB<T> {
            auto tmp = self.to_obj_coord(p);
            return shammath::AABB<T>(tmp.lower, tmp.upper);
        });
}

Register_pymod(pyshamrockctxinit) {

    shamlog_debug_ln("[Py]", "registering shamrock.Patch");

    py::class_<shamrock::patch::Patch>(m, "Patch")
        .def(py::init<>())
        .def_readwrite("id_patch", &shamrock::patch::Patch::id_patch)
        .def_readwrite("pack_node_index", &shamrock::patch::Patch::pack_node_index)
        .def_readwrite("load_value", &shamrock::patch::Patch::load_value)
        .def_readwrite("coord_min", &shamrock::patch::Patch::coord_min)
        .def_readwrite("coord_max", &shamrock::patch::Patch::coord_max)
        .def_readwrite("node_owner_id", &shamrock::patch::Patch::node_owner_id);

    register_patch_transform_class<f64_3>(m, "PatchCoordTransform_f64_3");
    register_patch_transform_class<f32_3>(m, "PatchCoordTransform_f32_3");
    register_patch_transform_class<i64_3>(m, "PatchCoordTransform_i64_3");

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
        .def(
            "collect_data",
            [](ShamrockCtx &ctx) {
                auto data = ctx.allgather_data();

                shamlog_info_ln("PatchScheduler", "collected :", data.size(), "patches");

                py::dict dic_out;

                using namespace shamrock;

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
            })
        .def("get_patch_list_global", &ShamrockCtx::get_patch_list_global);
}
