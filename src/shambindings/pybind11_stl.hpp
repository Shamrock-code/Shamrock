// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
#include <utility>
#include <type_traits>
#include <algorithm>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wkeyword-macro"
#define noexcept
#include <valarray>
#undef noexcept
#pragma GCC diagnostic pop
#endif

#include <pybind11/stl.h>
