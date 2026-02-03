// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HypergraphLog.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/HypergraphLog.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <unordered_map>
#include <memory>

namespace shamrock::solvergraph {

    std::unordered_map<u64, std::shared_ptr<INode>> inode_labels;

    void log_new_inode(u64 uuid) {
        shamcomm::logs::raw_ln("New inode created: ", uuid);
        inode_labels[uuid] = {};
    }

    void log_del_inode(u64 uuid) {
        shamcomm::logs::raw_ln("Inode deleted: ", uuid);
        inode_labels.erase(uuid);
    }

    void notify_inode_ptr(u64 uuid, std::shared_ptr<INode> &ptr) {
        if (!bool(inode_labels.at(uuid))) {
            if (inode_labels.at(uuid).get() != ptr.get()) {
                inode_labels[uuid] = ptr;
            }
        }
    }

} // namespace shamrock::solvergraph
