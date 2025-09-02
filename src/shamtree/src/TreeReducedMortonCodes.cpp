// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TreeReducedMortonCodes.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/TreeReducedMortonCodes.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shamtree/kernels/reduction_alg.hpp"

namespace shamrock::tree {

    template<class u_morton>
    void TreeReducedMortonCodes<u_morton>::build(
        sycl::queue &queue,
        u32 obj_cnt,
        u32 reduc_level,
        TreeMortonCodes<u_morton> &morton_codes,

        bool &one_cell_mode) {

        // return a sycl buffer from reduc index map instead
        shamlog_debug_sycl_ln(
            "RadixTree", "reduction algorithm"); // TODO put reduction level in class member

        // TODO document that the layout of reduc_index_map is in the end {0 .. ,i .. ,N ,0}
        // with the trailling 0 to invert the range for the walk in one cell mode

        reduction_alg(
            queue,
            obj_cnt,
            morton_codes.buf_morton,
            reduc_level,
            buf_reduc_index_map,
            tree_leaf_count);

        shamlog_debug_sycl_ln(
            "RadixTree",
            "reduction results : (before :",
            obj_cnt,
            " | after :",
            tree_leaf_count,
            ") ratio :",
            shambase::format_printf("%2.2f", f32(obj_cnt) / f32(tree_leaf_count)));

        if (tree_leaf_count > 1) {

            shamlog_debug_sycl_ln("RadixTree", "sycl_morton_remap_reduction");
            buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(tree_leaf_count);

            sycl_morton_remap_reduction(
                queue,
                tree_leaf_count,
                buf_reduc_index_map,
                morton_codes.buf_morton,
                buf_tree_morton);

            one_cell_mode = false;

        } else if (tree_leaf_count == 1) {

            tree_leaf_count = 2;
            one_cell_mode   = true;

            buf_tree_morton = std::make_unique<sycl::buffer<u_morton>>(
                shamalgs::memory::vector_to_buf(
                    shamsys::instance::get_compute_queue(), std::vector<u_morton>{0, 0})
                // tree morton = {0,0} is a flag for the one cell mode
            );

        } else {
            throw shambase::make_except_with_loc<std::runtime_error>("0 leaf tree cannot exists");
        }
    }

    template<class u_morton>
    bool TreeReducedMortonCodes<u_morton>::operator==(
        const TreeReducedMortonCodes<u_morton> &rhs) const {
        bool cmp = true;

        cmp = cmp && (tree_leaf_count == rhs.tree_leaf_count);

        using namespace shamalgs::primitives;

        cmp = cmp && (buf_reduc_index_map->size() == rhs.buf_reduc_index_map->size());

        cmp = cmp
              && equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_reduc_index_map,
                  *rhs.buf_reduc_index_map,
                  buf_reduc_index_map->size());
        cmp = cmp
              && equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_tree_morton,
                  *rhs.buf_tree_morton,
                  tree_leaf_count);

        return cmp;
    }

    template class TreeReducedMortonCodes<u32>;
    template class TreeReducedMortonCodes<u64>;

} // namespace shamrock::tree
