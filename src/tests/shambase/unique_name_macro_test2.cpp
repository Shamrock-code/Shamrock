// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/unique_name_macro.hpp"

static int __shamrock_unique_name(test_var) = 0;
static int __shamrock_unique_name(test_var) = 0;

static void __shamrock_unique_name(test_func)(){};
static void __shamrock_unique_name(test_func)(){};

// This file duplicates the content of unique_name_macro_test.cpp to test that using the same macro
// as the same spot does not provide linker errors
