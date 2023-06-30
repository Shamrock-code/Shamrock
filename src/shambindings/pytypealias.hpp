// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

#include "shambase/type_aliases.hpp"
#include "shambindings/pybindaliases.hpp"
#include <pybind11/stl.h>


namespace PYBIND11_NAMESPACE { namespace detail {




    template <> struct type_caster<f64_3> {
    public:
        PYBIND11_TYPE_CASTER(f64_3, const_name("f64_3"));

        bool load(handle src, bool) {
            PyObject *source = src.ptr();

            f64 x,y,z;

            if(!PyArg_ParseTuple(source, "ddd",&x,&y,&z)) {
                return false;
            }

            value = {x,y,z};

            return !PyErr_Occurred();
        }

        static handle cast(f64_3 src, return_value_policy /* policy */, handle /* parent */) {
            return Py_BuildValue("ddd", src.x(), src.y(), src.z());
        }
    };



    template <> struct type_caster<i32_3> {
    public:
        PYBIND11_TYPE_CASTER(i32_3, const_name("i32_3"));

        bool load(handle src, bool) {
            PyObject *source = src.ptr();

            i32 x,y,z;

            if(!PyArg_ParseTuple(source, "iii",&x,&y,&z)) {
                return false;
            }

            value = {x,y,z};

            return !PyErr_Occurred();
        }

        static handle cast(i32_3 src, return_value_policy /* policy */, handle /* parent */) {
            return Py_BuildValue("iii", src.x(), src.y(), src.z());
        }
    };



}} // namespace PYBIND11_NAMESPACE::detail

