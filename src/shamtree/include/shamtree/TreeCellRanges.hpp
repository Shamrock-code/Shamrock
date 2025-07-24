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
 * @file TreeCellRanges.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "kernels/compute_ranges.hpp"
#include "kernels/convert_ranges.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/sycl.hpp"
#include "shammath/sfc/morton.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/MortonKernels.hpp"
#include "shamtree/TreeReducedMortonCodes.hpp"
#include "shamtree/TreeStructure.hpp"

namespace shamrock::tree {

    template<class u_morton, class pos_t>
    class TreeCellRanges {

        using Morton             = shamrock::sfc::MortonCodes<u_morton, 3>;
        using ipos_t             = typename shamrock::sfc::MortonCodes<u_morton, 3>::int_vec_repr;
        static constexpr u32 dim = 3;

        public:
        // this one is not used, it should be removed
        std::unique_ptr<sycl::buffer<ipos_t>>
            buf_pos_min_cell; // size = total count //rename to ipos
        std::unique_ptr<sycl::buffer<ipos_t>> buf_pos_max_cell; // size = total count

        // optional
        std::unique_ptr<sycl::buffer<pos_t>>
            buf_pos_min_cell_flt; // size = total count //drop the flt part
        std::unique_ptr<sycl::buffer<pos_t>> buf_pos_max_cell_flt; // size = total count

        inline void build1(
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
                        tree_struct.internal_cell_count
                            + tree_reduced_morton_codes.tree_leaf_count);
                }
            }
        }

        void build2(sycl::queue &queue, u32 total_count, std::tuple<pos_t, pos_t> bounding_box) {

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

        inline bool are_range_int_built() {
            return bool(buf_pos_min_cell) && bool(buf_pos_max_cell);
        }

        inline bool are_range_float_built() {
            return bool(buf_pos_min_cell_flt) && bool(buf_pos_max_cell_flt);
        }

        inline TreeCellRanges() = default;

        inline TreeCellRanges(const TreeCellRanges &other)
            : buf_pos_min_cell(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_min_cell)), // size = total count
              buf_pos_max_cell(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_max_cell)), // size = total count
              buf_pos_min_cell_flt(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_min_cell_flt)), // size = total count
              buf_pos_max_cell_flt(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_pos_max_cell_flt)) // size = total count
        {}

        inline TreeCellRanges &operator=(TreeCellRanges &&other) noexcept {
            buf_pos_min_cell     = std::move(other.buf_pos_min_cell);
            buf_pos_max_cell     = std::move(other.buf_pos_max_cell);
            buf_pos_min_cell_flt = std::move(other.buf_pos_min_cell_flt);
            buf_pos_max_cell_flt = std::move(other.buf_pos_max_cell_flt);

            return *this;
        } // move assignment

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_pos_min_cell);
            add_ptr(buf_pos_max_cell);
            add_ptr(buf_pos_min_cell_flt);
            add_ptr(buf_pos_max_cell_flt);

            return sum;
        }

        inline friend bool operator==(const TreeCellRanges &t1, const TreeCellRanges &t2) {
            bool cmp = true;

            using namespace shamalgs::reduction;

            cmp = cmp
                  && equals_ptr(
                      shamsys::instance::get_compute_queue(),
                      t1.buf_pos_min_cell,
                      t2.buf_pos_min_cell);
            cmp = cmp
                  && equals_ptr(
                      shamsys::instance::get_compute_queue(),
                      t1.buf_pos_max_cell,
                      t2.buf_pos_max_cell);
            cmp = cmp
                  && equals_ptr(
                      shamsys::instance::get_compute_queue(),
                      t1.buf_pos_min_cell_flt,
                      t2.buf_pos_min_cell_flt);
            cmp = cmp
                  && equals_ptr(
                      shamsys::instance::get_compute_queue(),
                      t1.buf_pos_max_cell_flt,
                      t2.buf_pos_max_cell_flt);

            return cmp;
        }

        inline u32 get_total_tree_cell_count() {
            if (buf_pos_min_cell) {
                return buf_pos_min_cell->size();
            } else if (buf_pos_min_cell_flt) {
                return buf_pos_min_cell_flt->size();
            } else {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "no buffers are allocated");
            }
        }

        inline void serialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};
            u32 state = (bool(buf_pos_min_cell) ? 1 : 0) + (bool(buf_pos_min_cell_flt) ? 1 : 0) * 2;

            serializer.write(state);

            if (state == 1) {
                u32 sz = buf_pos_min_cell->size();
                serializer.write(sz);
                serializer.write_buf(*buf_pos_min_cell, sz);
                serializer.write_buf(*buf_pos_max_cell, sz);
            } else if (state == 2) {
                u32 sz = buf_pos_min_cell_flt->size();
                serializer.write(sz);
                serializer.write_buf(*buf_pos_min_cell_flt, sz);
                serializer.write_buf(*buf_pos_max_cell_flt, sz);
            } else if (state == 3) {
                u32 sz = buf_pos_min_cell->size();
                serializer.write(sz);
                serializer.write_buf(*buf_pos_min_cell, sz);
                serializer.write_buf(*buf_pos_max_cell, sz);
                serializer.write_buf(*buf_pos_min_cell_flt, sz);
                serializer.write_buf(*buf_pos_max_cell_flt, sz);
            }
        }

        inline shamalgs::SerializeSize serialize_byte_size() {

            using H = shamalgs::SerializeHelper;

            shamalgs::SerializeSize sum = H::serialize_byte_size<u32>();

            u32 state = (bool(buf_pos_min_cell) ? 1 : 0) + (bool(buf_pos_min_cell_flt) ? 1 : 0) * 2;

            if (state == 1) {
                u32 sz = buf_pos_min_cell->size();
                sum += H::serialize_byte_size<u32>();
                sum += H::serialize_byte_size<ipos_t>(sz);
                sum += H::serialize_byte_size<ipos_t>(sz);
            } else if (state == 2) {
                u32 sz = buf_pos_min_cell_flt->size();
                sum += H::serialize_byte_size<u32>();
                sum += H::serialize_byte_size<pos_t>(sz);
                sum += H::serialize_byte_size<pos_t>(sz);
            } else if (state == 3) {
                u32 sz = buf_pos_min_cell->size();
                sum += H::serialize_byte_size<u32>();
                sum += H::serialize_byte_size<ipos_t>(sz);
                sum += H::serialize_byte_size<ipos_t>(sz);
                sum += H::serialize_byte_size<pos_t>(sz);
                sum += H::serialize_byte_size<pos_t>(sz);
            }

            return sum;
        }

        inline static TreeCellRanges deserialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};

            TreeCellRanges ret;

            u32 state;
            serializer.load(state);

            if (state == 1) {
                u32 sz;
                serializer.load(sz);
                ret.buf_pos_min_cell = std::make_unique<sycl::buffer<ipos_t>>(sz);
                ret.buf_pos_max_cell = std::make_unique<sycl::buffer<ipos_t>>(sz);
                serializer.load_buf(*ret.buf_pos_min_cell, sz);
                serializer.load_buf(*ret.buf_pos_max_cell, sz);
            } else if (state == 2) {
                u32 sz;
                serializer.load(sz);
                ret.buf_pos_min_cell_flt = std::make_unique<sycl::buffer<pos_t>>(sz);
                ret.buf_pos_max_cell_flt = std::make_unique<sycl::buffer<pos_t>>(sz);
                serializer.load_buf(*ret.buf_pos_min_cell_flt, sz);
                serializer.load_buf(*ret.buf_pos_max_cell_flt, sz);
            } else if (state == 3) {
                u32 sz;
                serializer.load(sz);
                ret.buf_pos_min_cell     = std::make_unique<sycl::buffer<ipos_t>>(sz);
                ret.buf_pos_max_cell     = std::make_unique<sycl::buffer<ipos_t>>(sz);
                ret.buf_pos_min_cell_flt = std::make_unique<sycl::buffer<pos_t>>(sz);
                ret.buf_pos_max_cell_flt = std::make_unique<sycl::buffer<pos_t>>(sz);
                serializer.load_buf(*ret.buf_pos_min_cell, sz);
                serializer.load_buf(*ret.buf_pos_max_cell, sz);
                serializer.load_buf(*ret.buf_pos_min_cell_flt, sz);
                serializer.load_buf(*ret.buf_pos_max_cell_flt, sz);
            }

            return ret;
        }
    };

} // namespace shamrock::tree
