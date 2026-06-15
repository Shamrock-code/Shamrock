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
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <vector>

template<class T>
void register_field(py::module &m, const char *class_name) {
    using namespace shamrock::solvergraph;

    py::class_<Field<T>, IEdge>(m, class_name)
        .def(
            "get_buf",
            [](Field<T> &self, u64 id_patch) -> sham::DeviceBuffer<T> & {
                return self.get_buf(id_patch);
            },
            py::return_value_policy::reference)
        .def(
            "__repr__",
            [=](Field<T> &self) {
                return shambase::format(
                    "{}(label={}, tex_symbol={}, nvar={})",
                    class_name,
                    self.get_label(),
                    self.get_tex_symbol(),
                    self.get_nvar());
            })
        .def("collect_data", [](Field<T> &self) -> std::vector<T> {
            std::vector<T> base = {};
            self.get_refs().for_each([&](u64 id, std::reference_wrapper<PatchDataField<T>> &pdf) {
                auto copy = pdf.get().get_buf().copy_to_stdvec();
                base.insert(base.end(), copy.begin(), copy.end());
            });

            std::vector<T> collected = {};
            shamalgs::collective::vector_allgatherv(base, collected, MPI_COMM_WORLD);
            return collected;
        });

    std::string map_fields_name = []() -> std::string {
        if (std::is_same_v<T, f64>) {
            return "map_fields_f64";
        } else if (std::is_same_v<T, f64_3>) {
            return "map_fields_f64_3";
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>("Unsupported type");
        }
    }();

    m.def(
        map_fields_name.c_str(),
        [](py::function func,
           py::kwargs kwargs // only Field<T> are allowed
        ) {
            for (auto item : kwargs) {
                if (!py::isinstance<Field<T>>(item.second)) {
                    throw py::type_error(
                        "all keyword arguments to map_fields must be Field objects");
                }
            }

            shambase::DistributedData<u32> sizes = {};

            for (auto item : kwargs) {
                auto name = py::cast<std::string>(item.first);

                auto &field = py::cast<Field<T> &>(item.second);

                if (sizes.is_empty()) {
                    sizes = field.get_obj_cnts();
                } else {
                    field.check_sizes(sizes);
                }
            }

            Field<T> result = Field<T>(1, "ret", "ret");
            result.ensure_sizes(sizes);

            sizes.for_each([&](u64 id, u32 size) {
                py::dict call_kwargs;

                for (auto item : kwargs) {
                    auto name = py::cast<std::string>(item.first);

                    auto &field = py::cast<Field<T> &>(item.second);

                    auto vec_data = field.get(id).get_buf().copy_to_stdvec();
                    auto pyarray  = shamrock::VecToNumpy<T>::convert(vec_data);

                    call_kwargs[name.c_str()] = pyarray;
                }

                py::tuple args(1);
                args[0] = size;

                py::object py_result = func(*args, **call_kwargs);

                auto result_data = py_result.cast<std::vector<T>>();

                result.get(id).get_buf().copy_from_stdvec(result_data);
            });

            return result;
        });
}

ON_PYTHON_INIT {

    using namespace shamrock::solvergraph;

    py::class_<IEdge>(root_module, "IEdge")
        .def("get_label", &IEdge::get_label)
        .def("get_tex_symbol", &IEdge::get_tex_symbol);

    register_field<f64>(root_module, "Field_f64");
    register_field<f64_3>(root_module, "Field_f64_3");
}
