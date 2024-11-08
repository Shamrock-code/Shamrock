// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pybind11_stl.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#ifdef SHAMROCK_VALARRAY_FIX
    #include <type_traits>
    #include <algorithm>
    #include <utility>
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wkeyword-macro"
    #define noexcept
    #include <valarray>
    #undef noexcept
    #pragma GCC diagnostic pop
#endif

#include <pybind11/stl.h>
