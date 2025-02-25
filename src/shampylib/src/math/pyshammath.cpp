// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyshammath.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shampylib/math/pyAABB.hpp"
#include "shampylib/math/pyRay.hpp"

Register_pymod(pysham_mathinit) {

    py::module math_module = m.def_submodule("math", "Shamrock math lib");

    shampylib::init_shamrock_math_AABB<f64_3>(math_module, "AABB_f64_3");
    shampylib::init_shamrock_math_Ray<f64_3>(math_module, "Ray_f64_3");
}
