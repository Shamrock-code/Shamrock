// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyshammath.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/derivatives.hpp"
#include "shammath/matrix.hpp"
#include "shammath/matrix_op.hpp"
#include "shammath/paving_function.hpp"
#include "shampylib/math/pyAABB.hpp"
#include "shampylib/math/pyRay.hpp"
#include "shampylib/math/pySPHKernels.hpp"
#include "shampylib/math/pySfc.hpp"
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <functional>

Register_pymod(pysham_mathinit) {

    py::module math_module = m.def_submodule("math", "Shamrock math lib");

    shampylib::init_shamrock_math_AABB<f64_3>(math_module, "AABB_f64_3");
    shampylib::init_shamrock_math_Ray<f64_3>(math_module, "Ray_f64_3");
    shampylib::init_shamrock_math_sfc(math_module);
    shampylib::init_shamrock_math_sphkernels(math_module);

    math_module.def("derivative_upwind", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_upwind<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_centered", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_centered<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_3point_forward", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_3point_forward<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_3point_backward", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_3point_backward<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_5point_midpoint", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_5point_midpoint<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def(
        "estim_deriv_step",
        [](u32 order) {
            return shammath::estim_deriv_step<f64>(order);
        },
        R"pbdoc(
    Estim the correct step to use for a given order when using derivatives
    )pbdoc");

    py::class_<shammath::paving_function_periodic_3d<f64_3>>(
        math_module, "paving_function_periodic_3d")
        .def(py::init([](f64_3 box_size) {
            return std::make_unique<shammath::paving_function_periodic_3d<f64_3>>(
                shammath::paving_function_periodic_3d<f64_3>{box_size});
        }))
        .def("f", &shammath::paving_function_periodic_3d<f64_3>::f)
        .def("f_inv", &shammath::paving_function_periodic_3d<f64_3>::f_inv)
        .def("f_aabb", &shammath::paving_function_periodic_3d<f64_3>::f_aabb)
        .def("f_aabb_inv", &shammath::paving_function_periodic_3d<f64_3>::f_aabb_inv);

    py::class_<shammath::paving_function_general_3d<f64_3>>(
        math_module, "paving_function_general_3d")
        .def(
            py::init([](f64_3 box_size,
                        f64_3 box_center,
                        bool is_x_periodic,
                        bool is_y_periodic,
                        bool is_z_periodic) {
                return std::make_unique<shammath::paving_function_general_3d<f64_3>>(
                    shammath::paving_function_general_3d<f64_3>{
                        box_size, box_center, is_x_periodic, is_y_periodic, is_z_periodic});
            }))
        .def("f", &shammath::paving_function_general_3d<f64_3>::f)
        .def("f_inv", &shammath::paving_function_general_3d<f64_3>::f_inv)
        .def("f_aabb", &shammath::paving_function_general_3d<f64_3>::f_aabb)
        .def("f_aabb_inv", &shammath::paving_function_general_3d<f64_3>::f_aabb_inv);

    py::class_<shammath::paving_function_general_3d_shear_x<f64_3>>(
        math_module, "paving_function_general_3d_shear_x")
        .def(
            py::init([](f64_3 box_size,
                        f64_3 box_center,
                        bool is_x_periodic,
                        bool is_y_periodic,
                        bool is_z_periodic,
                        f64 shear_x) {
                return std::make_unique<shammath::paving_function_general_3d_shear_x<f64_3>>(
                    shammath::paving_function_general_3d_shear_x<f64_3>{
                        box_size,
                        box_center,
                        is_x_periodic,
                        is_y_periodic,
                        is_z_periodic,
                        shear_x});
            }))
        .def("f", &shammath::paving_function_general_3d_shear_x<f64_3>::f)
        .def("f_inv", &shammath::paving_function_general_3d_shear_x<f64_3>::f_inv)
        .def("f_aabb", &shammath::paving_function_general_3d_shear_x<f64_3>::f_aabb)
        .def("f_aabb_inv", &shammath::paving_function_general_3d_shear_x<f64_3>::f_aabb_inv);

    py::class_<f64_4x4>(math_module, "f64_4x4")
        .def(py::init([]() {
            return std::make_unique<f64_4x4>();
        }))
        .def(
            "__getitem__",
            [](const f64_4x4 &m, std::pair<int, int> idx) -> double {
                return m(idx.first, idx.second);
            })
        .def(
            "__setitem__",
            [](f64_4x4 &m, std::pair<int, int> idx, double value) {
                m(idx.first, idx.second) = value;
            })
        .def(
            "__repr__",
            [](const f64_4x4 &m) {
                std::ostringstream oss;
                oss << "[";
                for (size_t i = 0; i < 4; ++i) {
                    oss << "[";
                    for (size_t j = 0; j < 4; ++j) {
                        oss << m(i, j);
                        if (j + 1 < 4)
                            oss << ", ";
                    }
                    oss << "]";
                    if (i + 1 < 4)
                        oss << ",\n ";
                }
                oss << "]";
                return oss.str();
            })
        .def(
            "__mul__",
            [](const f64_4x4 &a, const f64_4x4 &b) {
                f64_4x4 ret;
                shammath::mat_prod(a.get_mdspan(), b.get_mdspan(), ret.get_mdspan());
                return ret;
            },
            py::is_operator())
        .def("to_pyarray", [](const f64_4x4 &self) {
            py::array_t<f64> ret({4, 4});
            for (u32 iy = 0; iy < 4; iy++) {
                for (u32 ix = 0; ix < 4; ix++) {
                    ret.mutable_at(ix, iy) = self(ix, iy);
                }
            }

            return ret;
        });

    math_module.def("get_identity_f64_4x4", []() -> f64_4x4 {
        return shammath::mat_identity<f64, 4>();
    });

    math_module.def("mat_mul", [](const f64_4x4 &a, const f64_4x4 &b) -> f64_4x4 {
        f64_4x4 ret;
        shammath::mat_prod(a.get_mdspan(), b.get_mdspan(), ret.get_mdspan());
        return ret;
    });

    math_module.def("mat_set_identity", [](f64_4x4 &a) {
        shammath::mat_set_identity(a.get_mdspan());
    });
}
