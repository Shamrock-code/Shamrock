// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyshammath.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
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
#include "shammath/symtensor_collections.hpp"
#include "shammath/symtensors.hpp"
#include "shampylib/math/pyAABB.hpp"
#include "shampylib/math/pyRay.hpp"
#include "shampylib/math/pySPHKernels.hpp"
#include "shampylib/math/pySfc.hpp"
#include <fmt/core.h>
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
        .def("f_aabb_inv", &shammath::paving_function_periodic_3d<f64_3>::f_aabb_inv)
        .def(
            "get_paving_index_intersecting",
            &shammath::paving_function_periodic_3d<f64_3>::get_paving_index_intersecting);

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
        .def("f_aabb_inv", &shammath::paving_function_general_3d<f64_3>::f_aabb_inv)
        .def(
            "get_paving_index_intersecting",
            &shammath::paving_function_general_3d<f64_3>::get_paving_index_intersecting);

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
        .def("f_aabb_inv", &shammath::paving_function_general_3d_shear_x<f64_3>::f_aabb_inv)
        .def(
            "get_paving_index_intersecting",
            &shammath::paving_function_general_3d_shear_x<f64_3>::get_paving_index_intersecting);

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
            "__matmul__",
            [](const f64_4x4 &a, const f64_4x4 &b) {
                f64_4x4 ret;
                shammath::mat_prod(a.get_mdspan(), b.get_mdspan(), ret.get_mdspan());
                return ret;
            },
            py::is_operator())
        .def("to_pyarray", [](const f64_4x4 &self) {
            py::array_t<f64> ret({4, 4});
            for (u32 i = 0; i < 4; i++) {
                for (u32 j = 0; j < 4; j++) {
                    ret.mutable_at(i, j) = self(i, j);
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

    // SymTensor3d_1 bindings
    py::class_<shammath::SymTensor3d_1<f64>>(math_module, "SymTensor3d_1_f64")
        .def(py::init<f64, f64, f64>(), py::arg("v_0"), py::arg("v_1"), py::arg("v_2"))
        .def(py::init<>())
        .def_readwrite("v_0", &shammath::SymTensor3d_1<f64>::v_0)
        .def_readwrite("v_1", &shammath::SymTensor3d_1<f64>::v_1)
        .def_readwrite("v_2", &shammath::SymTensor3d_1<f64>::v_2)
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_1<f64> &>(
                &shammath::SymTensor3d_1<f64>::inner, py::const_),
            "Inner product with another SymTensor3d_1")
        .def(
            "inner",
            py::overload_cast<const f64>(&shammath::SymTensor3d_1<f64>::inner, py::const_),
            "Scalar multiplication")
        .def("__mul__", &shammath::SymTensor3d_1<f64>::operator*, "Multiply by scalar")
        .def("__imul__", &shammath::SymTensor3d_1<f64>::operator*=, "In-place multiply by scalar")
        .def("__add__", &shammath::SymTensor3d_1<f64>::operator+, "Add two tensors")
        .def("__iadd__", &shammath::SymTensor3d_1<f64>::operator+=, "In-place add")
        .def("__sub__", &shammath::SymTensor3d_1<f64>::operator-, "Subtract two tensors")
        .def("__repr__", [](const shammath::SymTensor3d_1<f64> &t) {
            return fmt::format("SymTensor3d_1(v_0={}, v_1={}, v_2={})", t.v_0, t.v_1, t.v_2);
        });

    // SymTensor3d_2 bindings
    py::class_<shammath::SymTensor3d_2<f64>>(math_module, "SymTensor3d_2_f64")
        .def(
            py::init<f64, f64, f64, f64, f64, f64>(),
            py::arg("v_00"),
            py::arg("v_01"),
            py::arg("v_02"),
            py::arg("v_11"),
            py::arg("v_12"),
            py::arg("v_22"))
        .def(py::init<>())
        .def_readwrite("v_00", &shammath::SymTensor3d_2<f64>::v_00)
        .def_readwrite("v_01", &shammath::SymTensor3d_2<f64>::v_01)
        .def_readwrite("v_02", &shammath::SymTensor3d_2<f64>::v_02)
        .def_readwrite("v_11", &shammath::SymTensor3d_2<f64>::v_11)
        .def_readwrite("v_12", &shammath::SymTensor3d_2<f64>::v_12)
        .def_readwrite("v_22", &shammath::SymTensor3d_2<f64>::v_22)
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_2<f64> &>(
                &shammath::SymTensor3d_2<f64>::inner, py::const_),
            "Inner product with another SymTensor3d_2")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_1<f64> &>(
                &shammath::SymTensor3d_2<f64>::inner, py::const_),
            "Inner product with SymTensor3d_1")
        .def(
            "inner",
            py::overload_cast<const f64>(&shammath::SymTensor3d_2<f64>::inner, py::const_),
            "Scalar multiplication")
        .def("__mul__", &shammath::SymTensor3d_2<f64>::operator*, "Multiply by scalar")
        .def("__imul__", &shammath::SymTensor3d_2<f64>::operator*=, "In-place multiply by scalar")
        .def("__add__", &shammath::SymTensor3d_2<f64>::operator+, "Add two tensors")
        .def("__iadd__", &shammath::SymTensor3d_2<f64>::operator+=, "In-place add")
        .def("__sub__", &shammath::SymTensor3d_2<f64>::operator-, "Subtract two tensors")
        .def("__repr__", [](const shammath::SymTensor3d_2<f64> &t) {
            return fmt::format(
                "SymTensor3d_2(v_00={}, v_01={}, v_02={}, v_11={}, v_12={}, v_22={})",
                t.v_00,
                t.v_01,
                t.v_02,
                t.v_11,
                t.v_12,
                t.v_22);
        });

    // SymTensor3d_3 bindings
    py::class_<shammath::SymTensor3d_3<f64>>(math_module, "SymTensor3d_3_f64")
        .def(
            py::init<f64, f64, f64, f64, f64, f64, f64, f64, f64, f64>(),
            py::arg("v_000"),
            py::arg("v_001"),
            py::arg("v_002"),
            py::arg("v_011"),
            py::arg("v_012"),
            py::arg("v_022"),
            py::arg("v_111"),
            py::arg("v_112"),
            py::arg("v_122"),
            py::arg("v_222"))
        .def(py::init<>())
        .def_readwrite("v_000", &shammath::SymTensor3d_3<f64>::v_000)
        .def_readwrite("v_001", &shammath::SymTensor3d_3<f64>::v_001)
        .def_readwrite("v_002", &shammath::SymTensor3d_3<f64>::v_002)
        .def_readwrite("v_011", &shammath::SymTensor3d_3<f64>::v_011)
        .def_readwrite("v_012", &shammath::SymTensor3d_3<f64>::v_012)
        .def_readwrite("v_022", &shammath::SymTensor3d_3<f64>::v_022)
        .def_readwrite("v_111", &shammath::SymTensor3d_3<f64>::v_111)
        .def_readwrite("v_112", &shammath::SymTensor3d_3<f64>::v_112)
        .def_readwrite("v_122", &shammath::SymTensor3d_3<f64>::v_122)
        .def_readwrite("v_222", &shammath::SymTensor3d_3<f64>::v_222)
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_3<f64> &>(
                &shammath::SymTensor3d_3<f64>::inner, py::const_),
            "Inner product with another SymTensor3d_3")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_2<f64> &>(
                &shammath::SymTensor3d_3<f64>::inner, py::const_),
            "Inner product with SymTensor3d_2")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_1<f64> &>(
                &shammath::SymTensor3d_3<f64>::inner, py::const_),
            "Inner product with SymTensor3d_1")
        .def(
            "inner",
            py::overload_cast<const f64>(&shammath::SymTensor3d_3<f64>::inner, py::const_),
            "Scalar multiplication")
        .def("__mul__", &shammath::SymTensor3d_3<f64>::operator*, "Multiply by scalar")
        .def("__imul__", &shammath::SymTensor3d_3<f64>::operator*=, "In-place multiply by scalar")
        .def("__add__", &shammath::SymTensor3d_3<f64>::operator+, "Add two tensors")
        .def("__iadd__", &shammath::SymTensor3d_3<f64>::operator+=, "In-place add")
        .def("__sub__", &shammath::SymTensor3d_3<f64>::operator-, "Subtract two tensors")
        .def("__repr__", [](const shammath::SymTensor3d_3<f64> &t) {
            return fmt::format(
                "SymTensor3d_3(v_000={}, v_001={}, v_002={}, v_011={}, v_012={}, v_022={}, "
                "v_111={}, v_112={}, v_122={}, v_222={})",
                t.v_000,
                t.v_001,
                t.v_002,
                t.v_011,
                t.v_012,
                t.v_022,
                t.v_111,
                t.v_112,
                t.v_122,
                t.v_222);
        });

    // SymTensor3d_4 bindings
    py::class_<shammath::SymTensor3d_4<f64>>(math_module, "SymTensor3d_4_f64")
        .def(
            py::init<f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64>(),
            py::arg("v_0000"),
            py::arg("v_0001"),
            py::arg("v_0002"),
            py::arg("v_0011"),
            py::arg("v_0012"),
            py::arg("v_0022"),
            py::arg("v_0111"),
            py::arg("v_0112"),
            py::arg("v_0122"),
            py::arg("v_0222"),
            py::arg("v_1111"),
            py::arg("v_1112"),
            py::arg("v_1122"),
            py::arg("v_1222"),
            py::arg("v_2222"))
        .def(py::init<>())
        .def_readwrite("v_0000", &shammath::SymTensor3d_4<f64>::v_0000)
        .def_readwrite("v_0001", &shammath::SymTensor3d_4<f64>::v_0001)
        .def_readwrite("v_0002", &shammath::SymTensor3d_4<f64>::v_0002)
        .def_readwrite("v_0011", &shammath::SymTensor3d_4<f64>::v_0011)
        .def_readwrite("v_0012", &shammath::SymTensor3d_4<f64>::v_0012)
        .def_readwrite("v_0022", &shammath::SymTensor3d_4<f64>::v_0022)
        .def_readwrite("v_0111", &shammath::SymTensor3d_4<f64>::v_0111)
        .def_readwrite("v_0112", &shammath::SymTensor3d_4<f64>::v_0112)
        .def_readwrite("v_0122", &shammath::SymTensor3d_4<f64>::v_0122)
        .def_readwrite("v_0222", &shammath::SymTensor3d_4<f64>::v_0222)
        .def_readwrite("v_1111", &shammath::SymTensor3d_4<f64>::v_1111)
        .def_readwrite("v_1112", &shammath::SymTensor3d_4<f64>::v_1112)
        .def_readwrite("v_1122", &shammath::SymTensor3d_4<f64>::v_1122)
        .def_readwrite("v_1222", &shammath::SymTensor3d_4<f64>::v_1222)
        .def_readwrite("v_2222", &shammath::SymTensor3d_4<f64>::v_2222)
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_4<f64> &>(
                &shammath::SymTensor3d_4<f64>::inner, py::const_),
            "Inner product with another SymTensor3d_4")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_3<f64> &>(
                &shammath::SymTensor3d_4<f64>::inner, py::const_),
            "Inner product with SymTensor3d_3")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_2<f64> &>(
                &shammath::SymTensor3d_4<f64>::inner, py::const_),
            "Inner product with SymTensor3d_2")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_1<f64> &>(
                &shammath::SymTensor3d_4<f64>::inner, py::const_),
            "Inner product with SymTensor3d_1")
        .def(
            "inner",
            py::overload_cast<const f64>(&shammath::SymTensor3d_4<f64>::inner, py::const_),
            "Scalar multiplication")
        .def("__mul__", &shammath::SymTensor3d_4<f64>::operator*, "Multiply by scalar")
        .def("__imul__", &shammath::SymTensor3d_4<f64>::operator*=, "In-place multiply by scalar")
        .def("__add__", &shammath::SymTensor3d_4<f64>::operator+, "Add two tensors")
        .def("__iadd__", &shammath::SymTensor3d_4<f64>::operator+=, "In-place add")
        .def("__sub__", &shammath::SymTensor3d_4<f64>::operator-, "Subtract two tensors")
        .def("__repr__", [](const shammath::SymTensor3d_4<f64> &t) {
            return fmt::format(
                "SymTensor3d_4(v_0000={}, v_0001={}, v_0002={}, v_0011={}, v_0012={}, v_0022={}, "
                "v_0111={}, v_0112={}, v_0122={}, v_0222={}, v_1111={}, v_1112={}, v_1122={}, "
                "v_1222={}, v_2222={})",
                t.v_0000,
                t.v_0001,
                t.v_0002,
                t.v_0011,
                t.v_0012,
                t.v_0022,
                t.v_0111,
                t.v_0112,
                t.v_0122,
                t.v_0222,
                t.v_1111,
                t.v_1112,
                t.v_1122,
                t.v_1222,
                t.v_2222);
        });

    // SymTensor3d_5 bindings
    py::class_<shammath::SymTensor3d_5<f64>>(math_module, "SymTensor3d_5_f64")
        .def(
            py::init<
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64>(),
            py::arg("v_00000"),
            py::arg("v_00001"),
            py::arg("v_00002"),
            py::arg("v_00011"),
            py::arg("v_00012"),
            py::arg("v_00022"),
            py::arg("v_00111"),
            py::arg("v_00112"),
            py::arg("v_00122"),
            py::arg("v_00222"),
            py::arg("v_01111"),
            py::arg("v_01112"),
            py::arg("v_01122"),
            py::arg("v_01222"),
            py::arg("v_02222"),
            py::arg("v_11111"),
            py::arg("v_11112"),
            py::arg("v_11122"),
            py::arg("v_11222"),
            py::arg("v_12222"),
            py::arg("v_22222"))
        .def(py::init<>())
        .def_readwrite("v_00000", &shammath::SymTensor3d_5<f64>::v_00000)
        .def_readwrite("v_00001", &shammath::SymTensor3d_5<f64>::v_00001)
        .def_readwrite("v_00002", &shammath::SymTensor3d_5<f64>::v_00002)
        .def_readwrite("v_00011", &shammath::SymTensor3d_5<f64>::v_00011)
        .def_readwrite("v_00012", &shammath::SymTensor3d_5<f64>::v_00012)
        .def_readwrite("v_00022", &shammath::SymTensor3d_5<f64>::v_00022)
        .def_readwrite("v_00111", &shammath::SymTensor3d_5<f64>::v_00111)
        .def_readwrite("v_00112", &shammath::SymTensor3d_5<f64>::v_00112)
        .def_readwrite("v_00122", &shammath::SymTensor3d_5<f64>::v_00122)
        .def_readwrite("v_00222", &shammath::SymTensor3d_5<f64>::v_00222)
        .def_readwrite("v_01111", &shammath::SymTensor3d_5<f64>::v_01111)
        .def_readwrite("v_01112", &shammath::SymTensor3d_5<f64>::v_01112)
        .def_readwrite("v_01122", &shammath::SymTensor3d_5<f64>::v_01122)
        .def_readwrite("v_01222", &shammath::SymTensor3d_5<f64>::v_01222)
        .def_readwrite("v_02222", &shammath::SymTensor3d_5<f64>::v_02222)
        .def_readwrite("v_11111", &shammath::SymTensor3d_5<f64>::v_11111)
        .def_readwrite("v_11112", &shammath::SymTensor3d_5<f64>::v_11112)
        .def_readwrite("v_11122", &shammath::SymTensor3d_5<f64>::v_11122)
        .def_readwrite("v_11222", &shammath::SymTensor3d_5<f64>::v_11222)
        .def_readwrite("v_12222", &shammath::SymTensor3d_5<f64>::v_12222)
        .def_readwrite("v_22222", &shammath::SymTensor3d_5<f64>::v_22222)
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_5<f64> &>(
                &shammath::SymTensor3d_5<f64>::inner, py::const_),
            "Inner product with another SymTensor3d_5")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_4<f64> &>(
                &shammath::SymTensor3d_5<f64>::inner, py::const_),
            "Inner product with SymTensor3d_4")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_3<f64> &>(
                &shammath::SymTensor3d_5<f64>::inner, py::const_),
            "Inner product with SymTensor3d_3")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_2<f64> &>(
                &shammath::SymTensor3d_5<f64>::inner, py::const_),
            "Inner product with SymTensor3d_2")
        .def(
            "inner",
            py::overload_cast<const shammath::SymTensor3d_1<f64> &>(
                &shammath::SymTensor3d_5<f64>::inner, py::const_),
            "Inner product with SymTensor3d_1")
        .def(
            "inner",
            py::overload_cast<const f64>(&shammath::SymTensor3d_5<f64>::inner, py::const_),
            "Scalar multiplication")
        .def("__mul__", &shammath::SymTensor3d_5<f64>::operator*, "Multiply by scalar")
        .def("__imul__", &shammath::SymTensor3d_5<f64>::operator*=, "In-place multiply by scalar")
        .def("__add__", &shammath::SymTensor3d_5<f64>::operator+, "Add two tensors")
        .def("__iadd__", &shammath::SymTensor3d_5<f64>::operator+=, "In-place add")
        .def("__sub__", &shammath::SymTensor3d_5<f64>::operator-, "Subtract two tensors")
        .def("__repr__", [](const shammath::SymTensor3d_5<f64> &t) {
            return fmt::format(
                "SymTensor3d_5(v_00000={}, v_00001={}, v_00002={}, v_00011={}, v_00012={}, "
                "v_00022={}, v_00111={}, v_00112={}, v_00122={}, v_00222={}, v_01111={}, "
                "v_01112={}, v_01122={}, v_01222={}, v_02222={}, v_11111={}, v_11112={}, "
                "v_11122={}, v_11222={}, v_12222={}, v_22222={})",
                t.v_00000,
                t.v_00001,
                t.v_00002,
                t.v_00011,
                t.v_00012,
                t.v_00022,
                t.v_00111,
                t.v_00112,
                t.v_00122,
                t.v_00222,
                t.v_01111,
                t.v_01112,
                t.v_01122,
                t.v_01222,
                t.v_02222,
                t.v_11111,
                t.v_11112,
                t.v_11122,
                t.v_11222,
                t.v_12222,
                t.v_22222);
        });

    // SymTensorCollection bindings
    // SymTensorCollection<f64, 0, 5>
    py::class_<shammath::SymTensorCollection<f64, 0, 5>>(math_module, "SymTensorCollection_f64_0_5")
        .def(py::init<>())
        .def(
            py::init<
                f64,
                shammath::SymTensor3d_1<f64>,
                shammath::SymTensor3d_2<f64>,
                shammath::SymTensor3d_3<f64>,
                shammath::SymTensor3d_4<f64>,
                shammath::SymTensor3d_5<f64>>(),
            py::arg("t0"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t3"),
            py::arg("t4"),
            py::arg("t5"))
        .def_readwrite("t0", &shammath::SymTensorCollection<f64, 0, 5>::t0)
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 0, 5>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 0, 5>::t2)
        .def_readwrite("t3", &shammath::SymTensorCollection<f64, 0, 5>::t3)
        .def_readwrite("t4", &shammath::SymTensorCollection<f64, 0, 5>::t4)
        .def_readwrite("t5", &shammath::SymTensorCollection<f64, 0, 5>::t5)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 0, 5>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 0, 5>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 0, 5>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 0, 5>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 0, 5>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 0, 5> &c) {
            return fmt::format(
                "SymTensorCollection_f64_0_5(\n  t0={},\n  t1={},\n  t2={},\n  t3={},\n  t4={},\n  "
                "t5={}\n)",
                c.t0,
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>(),
                py::str(py::cast(c.t3)).cast<std::string>(),
                py::str(py::cast(c.t4)).cast<std::string>(),
                py::str(py::cast(c.t5)).cast<std::string>());
        });

    // SymTensorCollection<f64, 0, 4>
    py::class_<shammath::SymTensorCollection<f64, 0, 4>>(math_module, "SymTensorCollection_f64_0_4")
        .def(py::init<>())
        .def(
            py::init<
                f64,
                shammath::SymTensor3d_1<f64>,
                shammath::SymTensor3d_2<f64>,
                shammath::SymTensor3d_3<f64>,
                shammath::SymTensor3d_4<f64>>(),
            py::arg("t0"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t3"),
            py::arg("t4"))
        .def_readwrite("t0", &shammath::SymTensorCollection<f64, 0, 4>::t0)
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 0, 4>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 0, 4>::t2)
        .def_readwrite("t3", &shammath::SymTensorCollection<f64, 0, 4>::t3)
        .def_readwrite("t4", &shammath::SymTensorCollection<f64, 0, 4>::t4)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 0, 4>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 0, 4>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 0, 4>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 0, 4>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 0, 4>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 0, 4> &c) {
            return fmt::format(
                "SymTensorCollection_f64_0_4(\n  t0={},\n  t1={},\n  t2={},\n  t3={},\n  t4={}\n)",
                c.t0,
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>(),
                py::str(py::cast(c.t3)).cast<std::string>(),
                py::str(py::cast(c.t4)).cast<std::string>());
        });

    // SymTensorCollection<f64, 0, 3>
    py::class_<shammath::SymTensorCollection<f64, 0, 3>>(math_module, "SymTensorCollection_f64_0_3")
        .def(py::init<>())
        .def(
            py::init<
                f64,
                shammath::SymTensor3d_1<f64>,
                shammath::SymTensor3d_2<f64>,
                shammath::SymTensor3d_3<f64>>(),
            py::arg("t0"),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t3"))
        .def_readwrite("t0", &shammath::SymTensorCollection<f64, 0, 3>::t0)
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 0, 3>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 0, 3>::t2)
        .def_readwrite("t3", &shammath::SymTensorCollection<f64, 0, 3>::t3)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 0, 3>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 0, 3>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 0, 3>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 0, 3>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 0, 3>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 0, 3> &c) {
            return fmt::format(
                "SymTensorCollection_f64_0_3(\n  t0={},\n  t1={},\n  t2={},\n  t3={}\n)",
                c.t0,
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>(),
                py::str(py::cast(c.t3)).cast<std::string>());
        });

    // SymTensorCollection<f64, 0, 2>
    py::class_<shammath::SymTensorCollection<f64, 0, 2>>(math_module, "SymTensorCollection_f64_0_2")
        .def(py::init<>())
        .def(
            py::init<f64, shammath::SymTensor3d_1<f64>, shammath::SymTensor3d_2<f64>>(),
            py::arg("t0"),
            py::arg("t1"),
            py::arg("t2"))
        .def_readwrite("t0", &shammath::SymTensorCollection<f64, 0, 2>::t0)
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 0, 2>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 0, 2>::t2)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 0, 2>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 0, 2>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 0, 2>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 0, 2>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 0, 2>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 0, 2> &c) {
            return fmt::format(
                "SymTensorCollection_f64_0_2(\n  t0={},\n  t1={},\n  t2={}\n)",
                c.t0,
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>());
        });

    // SymTensorCollection<f64, 0, 1>
    py::class_<shammath::SymTensorCollection<f64, 0, 1>>(math_module, "SymTensorCollection_f64_0_1")
        .def(py::init<>())
        .def(py::init<f64, shammath::SymTensor3d_1<f64>>(), py::arg("t0"), py::arg("t1"))
        .def_readwrite("t0", &shammath::SymTensorCollection<f64, 0, 1>::t0)
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 0, 1>::t1)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 0, 1>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 0, 1>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 0, 1>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 0, 1>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 0, 1>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 0, 1> &c) {
            return fmt::format(
                "SymTensorCollection_f64_0_1(\n  t0={},\n  t1={}\n)",
                c.t0,
                py::str(py::cast(c.t1)).cast<std::string>());
        });

    // SymTensorCollection<f64, 0, 0>
    py::class_<shammath::SymTensorCollection<f64, 0, 0>>(math_module, "SymTensorCollection_f64_0_0")
        .def(py::init<>())
        .def(py::init<f64>(), py::arg("t0"))
        .def_readwrite("t0", &shammath::SymTensorCollection<f64, 0, 0>::t0)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 0, 0>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 0, 0>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 0, 0>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 0, 0>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 0, 0>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 0, 0> &c) {
            return fmt::format("SymTensorCollection_f64_0_0(t0={})", c.t0);
        });

    // SymTensorCollection<f64, 1, 5>
    py::class_<shammath::SymTensorCollection<f64, 1, 5>>(math_module, "SymTensorCollection_f64_1_5")
        .def(py::init<>())
        .def(
            py::init<
                shammath::SymTensor3d_1<f64>,
                shammath::SymTensor3d_2<f64>,
                shammath::SymTensor3d_3<f64>,
                shammath::SymTensor3d_4<f64>,
                shammath::SymTensor3d_5<f64>>(),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t3"),
            py::arg("t4"),
            py::arg("t5"))
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 1, 5>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 1, 5>::t2)
        .def_readwrite("t3", &shammath::SymTensorCollection<f64, 1, 5>::t3)
        .def_readwrite("t4", &shammath::SymTensorCollection<f64, 1, 5>::t4)
        .def_readwrite("t5", &shammath::SymTensorCollection<f64, 1, 5>::t5)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 1, 5>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 1, 5>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 1, 5>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 1, 5>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 1, 5>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 1, 5> &c) {
            return fmt::format(
                "SymTensorCollection_f64_1_5(\n  t1={},\n  t2={},\n  t3={},\n  t4={},\n  t5={}\n)",
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>(),
                py::str(py::cast(c.t3)).cast<std::string>(),
                py::str(py::cast(c.t4)).cast<std::string>(),
                py::str(py::cast(c.t5)).cast<std::string>());
        });

    // SymTensorCollection<f64, 1, 4>
    py::class_<shammath::SymTensorCollection<f64, 1, 4>>(math_module, "SymTensorCollection_f64_1_4")
        .def(py::init<>())
        .def(
            py::init<
                shammath::SymTensor3d_1<f64>,
                shammath::SymTensor3d_2<f64>,
                shammath::SymTensor3d_3<f64>,
                shammath::SymTensor3d_4<f64>>(),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t3"),
            py::arg("t4"))
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 1, 4>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 1, 4>::t2)
        .def_readwrite("t3", &shammath::SymTensorCollection<f64, 1, 4>::t3)
        .def_readwrite("t4", &shammath::SymTensorCollection<f64, 1, 4>::t4)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 1, 4>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 1, 4>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 1, 4>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 1, 4>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 1, 4>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 1, 4> &c) {
            return fmt::format(
                "SymTensorCollection_f64_1_4(\n  t1={},\n  t2={},\n  t3={},\n  t4={}\n)",
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>(),
                py::str(py::cast(c.t3)).cast<std::string>(),
                py::str(py::cast(c.t4)).cast<std::string>());
        });

    // SymTensorCollection<f64, 1, 3>
    py::class_<shammath::SymTensorCollection<f64, 1, 3>>(math_module, "SymTensorCollection_f64_1_3")
        .def(py::init<>())
        .def(
            py::init<
                shammath::SymTensor3d_1<f64>,
                shammath::SymTensor3d_2<f64>,
                shammath::SymTensor3d_3<f64>>(),
            py::arg("t1"),
            py::arg("t2"),
            py::arg("t3"))
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 1, 3>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 1, 3>::t2)
        .def_readwrite("t3", &shammath::SymTensorCollection<f64, 1, 3>::t3)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 1, 3>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 1, 3>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 1, 3>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 1, 3>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 1, 3>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 1, 3> &c) {
            return fmt::format(
                "SymTensorCollection_f64_1_3(\n  t1={},\n  t2={},\n  t3={}\n)",
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>(),
                py::str(py::cast(c.t3)).cast<std::string>());
        });

    // SymTensorCollection<f64, 1, 2>
    py::class_<shammath::SymTensorCollection<f64, 1, 2>>(math_module, "SymTensorCollection_f64_1_2")
        .def(py::init<>())
        .def(
            py::init<shammath::SymTensor3d_1<f64>, shammath::SymTensor3d_2<f64>>(),
            py::arg("t1"),
            py::arg("t2"))
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 1, 2>::t1)
        .def_readwrite("t2", &shammath::SymTensorCollection<f64, 1, 2>::t2)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 1, 2>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 1, 2>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 1, 2>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 1, 2>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 1, 2>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 1, 2> &c) {
            return fmt::format(
                "SymTensorCollection_f64_1_2(\n  t1={},\n  t2={}\n)",
                py::str(py::cast(c.t1)).cast<std::string>(),
                py::str(py::cast(c.t2)).cast<std::string>());
        });

    // SymTensorCollection<f64, 1, 1>
    py::class_<shammath::SymTensorCollection<f64, 1, 1>>(math_module, "SymTensorCollection_f64_1_1")
        .def(py::init<>())
        .def(py::init<shammath::SymTensor3d_1<f64>>(), py::arg("t1"))
        .def_readwrite("t1", &shammath::SymTensorCollection<f64, 1, 1>::t1)
        .def_static("zeros", &shammath::SymTensorCollection<f64, 1, 1>::zeros)
        .def_static("from_vec", &shammath::SymTensorCollection<f64, 1, 1>::from_vec, py::arg("v"))
        .def("__imul__", &shammath::SymTensorCollection<f64, 1, 1>::operator*=)
        .def("__iadd__", &shammath::SymTensorCollection<f64, 1, 1>::operator+=)
        .def("__sub__", &shammath::SymTensorCollection<f64, 1, 1>::operator-)
        .def("__repr__", [](const shammath::SymTensorCollection<f64, 1, 1> &c) {
            return fmt::format(
                "SymTensorCollection_f64_1_1(\n  t1={}\n)",
                py::str(py::cast(c.t1)).cast<std::string>());
        });
}
