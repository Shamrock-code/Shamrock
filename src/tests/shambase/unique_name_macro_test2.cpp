// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/unique_name_macro.hpp"

// this is an intentional duplicate of unique_name_macro_test.cpp
// to test case where the linker names could clash

static int __shamrock_unique_name(test_var) = 0;
static int __shamrock_unique_name(test_var) = 0;

static void __shamrock_unique_name(test_func)(){};
static void __shamrock_unique_name(test_func)(){};
