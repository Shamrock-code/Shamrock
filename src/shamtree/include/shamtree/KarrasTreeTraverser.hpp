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
 * @file KarrasTreeTraverser.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"

namespace shamtree {

    /**
     * @struct KarrasTreeTraverser
     * @brief Utility struct to traverse a Karras Radix Tree
     */
    struct KarrasTreeTraverser;

    /// read only accessor to buffer data
    struct KarrasTreeTraverserAccessed;

} // namespace shamtree

struct shamtree::KarrasTreeTraverserAccessed {
    const u32 *lchild_id;
    const u32 *rchild_id;
    const u8 *lchild_flag;
    const u8 *rchild_flag;
    u32 offset_leaf;

    /**
     * @brief Retrieves the left child node identifier for a given node ID.
     *
     * @param id The identifier of the node for which to find the left child.
     * @return The ID of the left child node, adjusted by the offset if the node is a leaf.
     */
    inline u32 get_left_child(u32 id) const {
        return lchild_id[id] + offset_leaf * u32(lchild_flag[id]);
    }

    /**
     * @brief Retrieves the right child node identifier for a given node ID.
     *
     * @param id The identifier of the node for which to find the right child.
     * @return The ID of the right child node, adjusted by the offset if the node is a leaf.
     */
    inline u32 get_right_child(u32 id) const {
        return rchild_id[id] + offset_leaf * u32(rchild_flag[id]);
    }

    /// is the given id a leaf
    inline bool is_id_leaf(u32 id) const { return id >= offset_leaf; }
};

struct shamtree::KarrasTreeTraverser {

    const sham::DeviceBuffer<u32> &buf_lchild_id;  ///< ref to left child id buffer
    const sham::DeviceBuffer<u32> &buf_rchild_id;  ///< ref to right child id buffer
    const sham::DeviceBuffer<u8> &buf_lchild_flag; ///< ref to left child flag buffer
    const sham::DeviceBuffer<u8> &buf_rchild_flag; ///< ref to right child flag buffer
    u32 offset_leaf; ///< how many internal nodes before the first leaf ?

    /// get read only accessor
    inline KarrasTreeTraverserAccessed get_read_access(sham::EventList &deps) const {
        return KarrasTreeTraverserAccessed{
            buf_lchild_id.get_read_access(deps),
            buf_rchild_id.get_read_access(deps),
            buf_lchild_flag.get_read_access(deps),
            buf_rchild_flag.get_read_access(deps),
            offset_leaf};
    }

    /// complete the buffer states with the resulting event
    inline void complete_event_state(sycl::event e) const {
        buf_lchild_id.complete_event_state(e);
        buf_rchild_id.complete_event_state(e);
        buf_lchild_flag.complete_event_state(e);
        buf_rchild_flag.complete_event_state(e);
    }
};
