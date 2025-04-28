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
 * @file KarrasRadixTree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"

namespace shamtree {
    class KarrasRadixTree;
}

class shamtree::KarrasRadixTree {

    public:
    inline u32 get_internal_cell_count() { return buf_lchild_id.get_size(); }

    inline u32 get_leaf_count() { return get_internal_cell_count() + 1; }

    sham::DeviceBuffer<u32> buf_lchild_id;  // size = internal
    sham::DeviceBuffer<u32> buf_rchild_id;  // size = internal
    sham::DeviceBuffer<u8> buf_lchild_flag; // size = internal
    sham::DeviceBuffer<u8> buf_rchild_flag; // size = internal
    sham::DeviceBuffer<u32> buf_endrange;   // size = internal

    KarrasRadixTree(
        sham::DeviceBuffer<u32> &&buf_lchild_id,
        sham::DeviceBuffer<u32> &&buf_rchild_id,
        sham::DeviceBuffer<u8> &&buf_lchild_flag,
        sham::DeviceBuffer<u8> &&buf_rchild_flag,
        sham::DeviceBuffer<u32> &&buf_endrange)
        : buf_lchild_id(std::move(buf_lchild_id)), buf_rchild_id(std::move(buf_rchild_id)),
          buf_lchild_flag(std::move(buf_lchild_flag)), buf_rchild_flag(std::move(buf_rchild_flag)),
          buf_endrange(std::move(buf_endrange)) {}
};

namespace shamtree {

    template<class Tmorton>
    KarrasRadixTree karras_tree_from_reduced_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &morton_codes,
        KarrasRadixTree &&recycled_tree);

    template<class Tmorton>
    KarrasRadixTree karras_tree_from_reduced_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &morton_codes);
} // namespace shamtree
