// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file IDataEdge.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shamdag/IDataEdge.hpp"
#include "shamdag/INode.hpp"
#include "shamdag/shamdag.hpp"

void IDataEdge::report_data_stealing() {
    if (auto spt = child.lock()) {
        shambase::get_check_ref(spt).report_data_stealing();
    }
}
