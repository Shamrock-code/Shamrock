// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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

#include "shammath/sfc/morton.hpp"
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

        void build1(
            sycl::queue &queue,
            TreeReducedMortonCodes<u_morton> &tree_reduced_morton_codes,
            TreeStructure<u_morton> &tree_struct);

        void build2(sycl::queue &queue, u32 total_count, std::tuple<pos_t, pos_t> bounding_box);

        inline bool are_range_int_built() {
            return bool(buf_pos_min_cell) && bool(buf_pos_max_cell);
        }

        inline bool are_range_float_built() {
            return bool(buf_pos_min_cell_flt) && bool(buf_pos_max_cell_flt);
        }

        inline TreeCellRanges() = default;

        TreeCellRanges(const TreeCellRanges &other);

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

        bool operator==(const TreeCellRanges &rhs) const;

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
