// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySolverGraph.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/MemPerfInfos.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

template<class T>
void register_field(py::module &m, const char *class_name) {
    using namespace shamrock::solvergraph;

    py::class_<Field<T>, IEdge>(m, class_name)
        .def(
            "get_buf",
            [](Field<f64> &self, u64 id_patch) -> sham::DeviceBuffer<f64> & {
                return self.get_buf(id_patch);
            },
            py::return_value_policy::reference)
        .def("__repr__", [=](Field<T> &self) {
            return shambase::format(
                "{}(label={}, tex_symbol={}, nvar={})",
                class_name,
                self.get_label(),
                self.get_tex_symbol(),
                self.get_nvar());
        });
}

Register_pymod(shamsolvergraphinit) {

    using namespace shamrock::solvergraph;

    py::class_<IEdge>(m, "IEdge")
        .def("get_label", &IEdge::get_label)
        .def("get_tex_symbol", &IEdge::get_tex_symbol);

    register_field<f64>(m, "Field_f64");
    register_field<f64_3>(m, "Field_f64_3");
}
