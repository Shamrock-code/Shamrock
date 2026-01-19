// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAABB.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/AABB.hpp"

namespace shampylib {

    template<class T>
    void init_shamrock_math_AABB(py::module &m, std::string name) {
        py::class_<shammath::AABB<T>>(m, name.c_str())
            .def(py::init([](f64_3 min, f64_3 max) {
                return std::make_unique<shammath::AABB<T>>(min, max);
            }))
            .def("get_intersect", &shammath::AABB<T>::get_intersect)
            .def("get_volume", &shammath::AABB<T>::get_volume)
            .def("get_center", &shammath::AABB<T>::get_center)
            .def("sum_bounds", &shammath::AABB<T>::sum_bounds)
            .def_readwrite(
                "lower", &shammath::AABB<T>::lower, py::return_value_policy::reference_internal)
            .def_readwrite(
                "upper", &shammath::AABB<T>::upper, py::return_value_policy::reference_internal)
            .def("delt", &shammath::AABB<T>::delt)
            .def("is_not_empty", &shammath::AABB<T>::is_not_empty)
            .def("is_volume_not_null", &shammath::AABB<T>::is_volume_not_null)
            .def("is_surface", &shammath::AABB<T>::is_surface)
            .def("is_surface_or_volume", &shammath::AABB<T>::is_surface_or_volume)
            .def("intersect_ray", &shammath::AABB<T>::intersect_ray)
            .def(
                "__str__",
                [](const shammath::AABB<T> &aabb) {
                    return shambase::format("AABB(lower={}, upper={})", aabb.lower, aabb.upper);
                })
            .def("__repr__", [](const shammath::AABB<T> &aabb) {
                return shambase::format("AABB(lower={}, upper={})", aabb.lower, aabb.upper);
            });
    }

    template void init_shamrock_math_AABB<f64_3>(py::module &m, std::string name);

} // namespace shampylib
