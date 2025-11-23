// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file HypergraphLog.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include <memory>

namespace shamrock::solvergraph {

    class INode; // forward declaration

    void log_new_inode(u64 uuid);
    void log_del_inode(u64 uuid);

    void notify_inode_ptr(u64 uuid, std::shared_ptr<INode> &ptr);

} // namespace shamrock::solvergraph
