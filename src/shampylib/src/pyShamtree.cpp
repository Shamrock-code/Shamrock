// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamtree.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include <pybind11/complex.h>

template<class Tmorton, class Tvec, u32 dim>
inline void register_CLBVH(py::module &m, const char *class_name) {

    using CLBVH = shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>;

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.tree." + std::string(class_name));
    py::class_<CLBVH>(m, class_name)
        .def(py::init([]() {
            return std::make_unique<CLBVH>(
                CLBVH::make_empty(shamsys::instance::get_compute_scheduler_ptr()));
        }))
        .def(
            "rebuild_from_positions",
            [](CLBVH &self,
               sham::DeviceBuffer<Tvec> &positions,
               const shammath::AABB<Tvec> &bounding_box,
               u32 compression_level) {
                self.rebuild_from_positions(positions, bounding_box, compression_level);
            })
        .def(
            "get_leaf_cell_count",
            [](CLBVH &self) {
                return self.get_leaf_cell_count();
            })
        .def(
            "get_internal_cell_count",
            [](CLBVH &self) {
                return self.get_internal_cell_count();
            })
        .def("get_total_cell_count", [](CLBVH &self) {
            return self.get_total_cell_count();
        });
}

template<class Tmorton, class Tvec, u32 dim>
inline void register_dtt_alg(py::module &m) {
    py::class_<shamtree::DTTResult>(m, "DTTResult").def(py::init([]() {
        return std::make_unique<shamtree::DTTResult>(shamtree::DTTResult{
            sham::DeviceBuffer<u32_2>(0, shamsys::instance::get_compute_scheduler_ptr()),
            sham::DeviceBuffer<u32_2>(0, shamsys::instance::get_compute_scheduler_ptr())});
    }));

    m.def(
        "clbvh_dual_tree_traversal",
        [](shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
           shambase::VecComponent<Tvec> theta_crit) {
            return shamtree::clbvh_dual_tree_traversal(
                shamsys::instance::get_compute_scheduler_ptr(), bvh, theta_crit);
        });

    m.def(
        "benchmark_clbvh_dual_tree_traversal",
        [](shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
           shambase::VecComponent<Tvec> theta_crit) {
            shambase::Timer t;
            t.start();
            shamtree::clbvh_dual_tree_traversal(
                shamsys::instance::get_compute_scheduler_ptr(), bvh, theta_crit);
            t.end();
            return t.elasped_sec();
        });

    m.def("get_default_impl_list_clbvh_dual_tree_traversal", []() {
        return shamtree::impl::get_default_impl_list_clbvh_dual_tree_traversal();
    });

    m.def(
        "set_impl_clbvh_dual_tree_traversal",
        [](const std::string &impl, const std::string &param = "") {
            shamtree::impl::set_impl_clbvh_dual_tree_traversal(impl, param);
        });

    m.def("get_current_impl_clbvh_dual_tree_traversal_impl", []() {
        return shamtree::impl::get_current_impl_clbvh_dual_tree_traversal_impl();
    });
}

Register_pymod(shamtreelibinit) {

    py::module shamtree_module = m.def_submodule("tree", "backend library");

    register_CLBVH<u64, f64_3, 3>(shamtree_module, "CLBVH_u64_f64_3");
    register_dtt_alg<u64, f64_3, 3>(shamtree_module);
}
