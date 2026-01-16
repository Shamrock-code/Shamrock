// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TreeCellRanges.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/TreeCellRanges.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamtree/MortonKernels.hpp"
#include "shamtree/kernels/compute_ranges.hpp"

namespace shamrock::tree {

    template<class u_morton, class pos_t>
    void TreeCellRanges<u_morton, pos_t>::build1(
        sycl::queue &queue,
        TreeReducedMortonCodes<u_morton> &tree_reduced_morton_codes,
        TreeStructure<u_morton> &tree_struct) {
        if (!tree_struct.one_cell_mode) {

            shamlog_debug_sycl_ln("RadixTree", "compute_cellvolume");

            buf_pos_min_cell = std::make_unique<sycl::buffer<ipos_t>>(
                tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);
            buf_pos_max_cell = std::make_unique<sycl::buffer<ipos_t>>(
                tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);

            sycl_compute_cell_ranges(
                queue,
                tree_reduced_morton_codes.tree_leaf_count,
                tree_struct.internal_cell_count,
                tree_reduced_morton_codes.buf_tree_morton,
                tree_struct.buf_lchild_id,
                tree_struct.buf_rchild_id,
                tree_struct.buf_lchild_flag,
                tree_struct.buf_rchild_flag,
                tree_struct.buf_endrange,
                buf_pos_min_cell,
                buf_pos_max_cell);

        } else {
            // throw shamrock_exc("one cell mode is not implemented");
            // TODO do some extensive test on one cell mode

            buf_pos_min_cell = std::make_unique<sycl::buffer<ipos_t>>(
                tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);
            buf_pos_max_cell = std::make_unique<sycl::buffer<ipos_t>>(
                tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);

            {

                sycl::host_accessor pos_min_cell{
                    *buf_pos_min_cell, sycl::write_only, sycl::no_init};
                sycl::host_accessor pos_max_cell{
                    *buf_pos_max_cell, sycl::write_only, sycl::no_init};

                pos_min_cell[0] = {0, 0, 0};
                pos_max_cell[0] = {Morton::max_val, Morton::max_val, Morton::max_val};

                pos_min_cell[1] = {0, 0, 0};
                pos_max_cell[1] = {Morton::max_val, Morton::max_val, Morton::max_val};

                pos_min_cell[2] = {0, 0, 0};
                pos_max_cell[2] = {0, 0, 0};

                shamlog_debug_sycl_ln("RadixTree", "compute_cellvolume one cell mode");
                shamlog_debug_sycl_ln(
                    "RadixTree",
                    " -> ",
                    pos_min_cell[0],
                    pos_max_cell[0],
                    pos_min_cell[1],
                    pos_max_cell[1],
                    pos_min_cell[2],
                    pos_max_cell[2],
                    "len =",
                    tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);
            }
        }
    }

    template<class u_morton, class pos_t>
    void TreeCellRanges<u_morton, pos_t>::build2(
        sycl::queue &queue, u32 total_count, std::tuple<pos_t, pos_t> bounding_box) {

        buf_pos_min_cell_flt = std::make_unique<sycl::buffer<pos_t>>(total_count);
        buf_pos_max_cell_flt = std::make_unique<sycl::buffer<pos_t>>(total_count);

        shamlog_debug_sycl_ln("RadixTree", "sycl_convert_cell_range");

        shamrock::sfc::MortonKernels<u_morton, pos_t, dim>::sycl_irange_to_range(
            queue,
            total_count,
            std::get<0>(bounding_box),
            std::get<1>(bounding_box),
            buf_pos_min_cell,
            buf_pos_max_cell,
            buf_pos_min_cell_flt,
            buf_pos_max_cell_flt);

        // remove old buf ?
    }

    template<class u_morton, class pos_t>
    TreeCellRanges<u_morton, pos_t>::TreeCellRanges(const TreeCellRanges<u_morton, pos_t> &other)
        : buf_pos_min_cell(
              shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_min_cell)), // size = total count
          buf_pos_max_cell(
              shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_max_cell)), // size = total count
          buf_pos_min_cell_flt(
              shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_min_cell_flt)), // size = total count
          buf_pos_max_cell_flt(
              shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_max_cell_flt)) // size = total count
    {}

    template<class u_morton, class pos_t>
    bool TreeCellRanges<u_morton, pos_t>::operator==(
        const TreeCellRanges<u_morton, pos_t> &rhs) const {
        bool cmp = true;

        using namespace shamalgs::primitives;

        cmp = cmp
              && equals_ptr(
                  shamsys::instance::get_compute_queue(), buf_pos_min_cell, rhs.buf_pos_min_cell);
        cmp = cmp
              && equals_ptr(
                  shamsys::instance::get_compute_queue(), buf_pos_max_cell, rhs.buf_pos_max_cell);
        cmp = cmp
              && equals_ptr(
                  shamsys::instance::get_compute_queue(),
                  buf_pos_min_cell_flt,
                  rhs.buf_pos_min_cell_flt);
        cmp = cmp
              && equals_ptr(
                  shamsys::instance::get_compute_queue(),
                  buf_pos_max_cell_flt,
                  rhs.buf_pos_max_cell_flt);

        return cmp;
    }

    template class TreeCellRanges<u32, f64_3>;
    template class TreeCellRanges<u64, f64_3>;
    template class TreeCellRanges<u32, f32_3>;
    template class TreeCellRanges<u64, f32_3>;
    template class TreeCellRanges<u32, i64_3>;
    template class TreeCellRanges<u64, i64_3>;
    template class TreeCellRanges<u32, u64_3>;
    template class TreeCellRanges<u64, u64_3>;
    template class TreeCellRanges<u32, u32_3>;
    template class TreeCellRanges<u64, u32_3>;

} // namespace shamrock::tree
