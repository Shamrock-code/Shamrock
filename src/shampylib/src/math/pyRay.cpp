// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyRay.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/AABB.hpp"

namespace shampylib {

    template<class T>
    void init_shamrock_math_Ray(py::module &m, std::string name) {
        py::class_<shammath::Ray<T>>(m, name.c_str())
            .def(py::init([](f64_3 origin, f64_3 direction) {
                return std::make_unique<shammath::Ray<T>>(origin, direction);
            }))
            .def(
                "origin",
                [](shammath::Ray<T> &ray) {
                    return ray.origin;
                })
            .def(
                "direction",
                [](shammath::Ray<T> &ray) {
                    return ray.direction;
                })
            .def("inv_direction", [](shammath::Ray<T> &ray) {
                return ray.inv_direction;
            });
    }

    template void init_shamrock_math_Ray<f64_3>(py::module &m, std::string name);

    template<class T>
    void init_shamrock_math_RingRay(py::module &m, std::string name) {
        py::class_<shammath::RingRay<T>>(m, name.c_str())
            .def(py::init([](f64_3 center, f64 radius, f64_3 e_x, f64_3 e_y) {
                return std::make_unique<shammath::RingRay<T>>(center, radius, e_x, e_y);
            }))
            .def(
                "center",
                [](shammath::RingRay<T> &ring_ray) {
                    return ring_ray.center;
                })
            .def(
                "radius",
                [](shammath::RingRay<T> &ring_ray) {
                    return ring_ray.radius;
                })
            .def(
                "e_x",
                [](shammath::RingRay<T> &ring_ray) {
                    return ring_ray.e_x;
                })
            .def("e_y", [](shammath::RingRay<T> &ring_ray) {
                return ring_ray.e_y;
            });
    }

    template void init_shamrock_math_RingRay<f64_3>(py::module &m, std::string name);

} // namespace shampylib
