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
 * @file TreeReducedMortonCodes.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "kernels/reduction_alg.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/reduction.hpp"
#include "shammath/CoordRange.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/RadixTreeMortonBuilder.hpp"
#include "shamtree/TreeMortonCodes.hpp"

namespace shamrock::tree {

    template<class u_morton>
    class TreeReducedMortonCodes {
        public:
        u32 tree_leaf_count;
        std::unique_ptr<sycl::buffer<u32>> buf_reduc_index_map;
        std::unique_ptr<sycl::buffer<u_morton>> buf_tree_morton; // size = leaf cnt

        inline void build(
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
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "0 leaf tree cannot exists");
            }
        }

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            sum += sizeof(tree_leaf_count);

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_reduc_index_map);
            add_ptr(buf_tree_morton);

            return sum;
        }

        inline TreeReducedMortonCodes() = default;

        inline TreeReducedMortonCodes(const TreeReducedMortonCodes &other)
            : tree_leaf_count(other.tree_leaf_count),
              buf_reduc_index_map(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(), other.buf_reduc_index_map)),
              buf_tree_morton(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(), other.buf_tree_morton)) {}

        inline TreeReducedMortonCodes &operator=(TreeReducedMortonCodes &&other) noexcept {
            tree_leaf_count     = std::move(other.tree_leaf_count);
            buf_reduc_index_map = std::move(other.buf_reduc_index_map);
            buf_tree_morton     = std::move(other.buf_tree_morton);

            return *this;
        } // move assignment

        inline friend bool
        operator==(const TreeReducedMortonCodes &t1, const TreeReducedMortonCodes &t2) {
            bool cmp = true;

            cmp = cmp && (t1.tree_leaf_count == t2.tree_leaf_count);

            using namespace shamalgs::reduction;

            cmp = cmp && (t1.buf_reduc_index_map->size() == t2.buf_reduc_index_map->size());

            cmp = cmp
                  && equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_reduc_index_map,
                      *t2.buf_reduc_index_map,
                      t1.buf_reduc_index_map->size());
            cmp = cmp
                  && equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_tree_morton,
                      *t2.buf_tree_morton,
                      t1.tree_leaf_count);

            return cmp;
        }

        /**
         * @brief serialize a TreeMortonCodes object
         *
         * @param serializer
         */
        inline void serialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};

            serializer.write(tree_leaf_count);
            if (!buf_reduc_index_map) {
                throw shambase::make_except_with_loc<std::runtime_error>("missing buffer");
            }
            serializer.write<u32>(buf_reduc_index_map->size());
            serializer.write_buf(*buf_reduc_index_map, buf_reduc_index_map->size());
            if (!buf_tree_morton) {
                throw shambase::make_except_with_loc<std::runtime_error>("missing buffer");
            }
            serializer.write_buf(*buf_tree_morton, tree_leaf_count);
        }

        inline static TreeReducedMortonCodes deserialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};

            TreeReducedMortonCodes ret;
            serializer.load(ret.tree_leaf_count);

            u32 tmp;
            serializer.load(tmp);

            ret.buf_reduc_index_map = std::make_unique<sycl::buffer<u32>>(tmp);
            ret.buf_tree_morton     = std::make_unique<sycl::buffer<u_morton>>(ret.tree_leaf_count);

            serializer.load_buf(*ret.buf_reduc_index_map, tmp);
            serializer.load_buf(*ret.buf_tree_morton, ret.tree_leaf_count);

            return ret;
        }

        inline shamalgs::SerializeSize serialize_byte_size() {

            using H = shamalgs::SerializeHelper;

            return H::serialize_byte_size<u32>() * 2
                   + H::serialize_byte_size<u32>(buf_reduc_index_map->size())
                   + H::serialize_byte_size<u_morton>(tree_leaf_count);
        }
    };

} // namespace shamrock::tree
