// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pytypealias.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"

namespace PYBIND11_NAMESPACE {
    namespace detail {

        template<>
        struct type_caster<f64_3> {
            public:
            PYBIND11_TYPE_CASTER(f64_3, const_name("f64_3"));

            bool load(handle src, bool) {
                PyObject *source = src.ptr();

                f64 x, y, z;

                if (!PyArg_ParseTuple(source, "ddd", &x, &y, &z)) {
                    return false;
                }

                value = {x, y, z};

                return !PyErr_Occurred();
            }

            static handle cast(f64_3 src, return_value_policy /* policy */, handle /* parent */) {
                return Py_BuildValue("ddd", src.x(), src.y(), src.z());
            }
        };

        template<>
        struct type_caster<i32_3> {
            public:
            PYBIND11_TYPE_CASTER(i32_3, const_name("i32_3"));

            bool load(handle src, bool) {
                PyObject *source = src.ptr();

                i32 x, y, z;

                if (!PyArg_ParseTuple(source, "iii", &x, &y, &z)) {
                    return false;
                }

                value = {x, y, z};

                return !PyErr_Occurred();
            }

            static handle cast(i32_3 src, return_value_policy /* policy */, handle /* parent */) {
                return Py_BuildValue("iii", src.x(), src.y(), src.z());
            }
        };

        template<>
        struct type_caster<i64_3> {
            public:
            PYBIND11_TYPE_CASTER(i64_3, const_name("i64_3"));

            bool load(handle src, bool) {
                PyObject *source = src.ptr();

                i64 x, y, z;

                if (!PyArg_ParseTuple(source, "LLL", &x, &y, &z)) {
                    return false;
                }

                value = {x, y, z};

                return !PyErr_Occurred();
            }

            static handle cast(i64_3 src, return_value_policy /* policy */, handle /* parent */) {
                return Py_BuildValue("LLL", src.x(), src.y(), src.z());
            }
        };

        template<>
        struct type_caster<u32_3> {
            public:
            PYBIND11_TYPE_CASTER(u32_3, const_name("u32_3"));

            bool load(handle src, bool) {
                PyObject *source = src.ptr();

                u32 x, y, z;

                if (!PyArg_ParseTuple(source, "III", &x, &y, &z)) {
                    return false;
                }

                value = {x, y, z};

                return !PyErr_Occurred();
            }

            static handle cast(u32_3 src, return_value_policy /* policy */, handle /* parent */) {
                return Py_BuildValue("III", src.x(), src.y(), src.z());
            }
        };

        template<>
        struct type_caster<u64_3> {
            public:
            PYBIND11_TYPE_CASTER(u64_3, const_name("u64_3"));

            bool load(handle src, bool) {
                PyObject *source = src.ptr();

                i64 x, y, z;

                if (!PyArg_ParseTuple(source, "KKK", &x, &y, &z)) {
                    return false;
                }

                value = {x, y, z};

                return !PyErr_Occurred();
            }

            static handle cast(u64_3 src, return_value_policy /* policy */, handle /* parent */) {
                return Py_BuildValue("KKK", src.x(), src.y(), src.z());
            }
        };

    } // namespace detail
} // namespace PYBIND11_NAMESPACE
