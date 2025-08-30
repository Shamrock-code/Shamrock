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
 * @file TreeStructure.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamalgs/serialize.hpp"

namespace shamrock::tree {

    template<class u_morton>
    class TreeStructure {

        public:
        u32 internal_cell_count;
        bool one_cell_mode = false;

        std::unique_ptr<sycl::buffer<u32>> buf_lchild_id;  // size = internal
        std::unique_ptr<sycl::buffer<u32>> buf_rchild_id;  // size = internal
        std::unique_ptr<sycl::buffer<u8>> buf_lchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u8>> buf_rchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u32>> buf_endrange;   // size = internal

        bool is_built() {
            return bool(buf_lchild_id) && bool(buf_rchild_id) && bool(buf_lchild_flag)
                   && bool(buf_rchild_flag) && bool(buf_endrange);
        }

        void build(
            sycl::queue &queue, u32 _internal_cell_count, sycl::buffer<u_morton> &morton_buf);

        void build_one_cell_mode();

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;
            sum += sizeof(internal_cell_count);
            sum += sizeof(one_cell_mode);

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_lchild_id);
            add_ptr(buf_rchild_id);
            add_ptr(buf_lchild_flag);
            add_ptr(buf_rchild_flag);
            add_ptr(buf_endrange);

            return sum;
        }

        bool operator==(const TreeStructure &rhs) const;

        inline TreeStructure() = default;

        TreeStructure(const TreeStructure &other);

        inline TreeStructure &operator=(TreeStructure &&other) noexcept {
            internal_cell_count = std::move(other.internal_cell_count);
            one_cell_mode       = std::move(other.one_cell_mode);
            buf_lchild_id       = std::move(other.buf_lchild_id);
            buf_rchild_id       = std::move(other.buf_rchild_id);
            buf_lchild_flag     = std::move(other.buf_lchild_flag);
            buf_rchild_flag     = std::move(other.buf_rchild_flag);
            buf_endrange        = std::move(other.buf_endrange);

            return *this;
        } // move assignment

        inline TreeStructure(
            u32 internal_cell_count,
            bool one_cell_mode,
            std::unique_ptr<sycl::buffer<u32>> &&buf_lchild_id,
            std::unique_ptr<sycl::buffer<u32>> &&buf_rchild_id,
            std::unique_ptr<sycl::buffer<u8>> &&buf_lchild_flag,
            std::unique_ptr<sycl::buffer<u8>> &&buf_rchild_flag,
            std::unique_ptr<sycl::buffer<u32>> &&buf_endrange)
            : internal_cell_count(internal_cell_count), one_cell_mode(one_cell_mode),
              buf_lchild_id(std::move(buf_lchild_id)), buf_rchild_id(std::move(buf_rchild_id)),
              buf_lchild_flag(std::move(buf_lchild_flag)),
              buf_rchild_flag(std::move(buf_rchild_flag)), buf_endrange(std::move(buf_endrange)) {}

        inline void serialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};
            serializer.write(internal_cell_count);
            serializer.write((one_cell_mode) ? 1_u32 : 0_u32);
            serializer.write_buf(*buf_lchild_id, internal_cell_count);
            serializer.write_buf(*buf_rchild_id, internal_cell_count);
            serializer.write_buf(*buf_lchild_flag, internal_cell_count);
            serializer.write_buf(*buf_rchild_flag, internal_cell_count);
            serializer.write_buf(*buf_endrange, internal_cell_count);
        }

        inline shamalgs::SerializeSize serialize_byte_size() {

            using H = shamalgs::SerializeHelper;

            return H::serialize_byte_size<u32>() + H::serialize_byte_size<u32>()
                   + H::serialize_byte_size<u32>(internal_cell_count)
                   + H::serialize_byte_size<u32>(internal_cell_count)
                   + H::serialize_byte_size<u8>(internal_cell_count)
                   + H::serialize_byte_size<u8>(internal_cell_count)
                   + H::serialize_byte_size<u32>(internal_cell_count);
        }

        inline static TreeStructure deserialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};
            TreeStructure strc;

            serializer.load(strc.internal_cell_count);
            u32 one_cell;

            serializer.load(one_cell);
            strc.one_cell_mode = (one_cell == 1);

            strc.buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(strc.internal_cell_count);
            strc.buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(strc.internal_cell_count);
            strc.buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(strc.internal_cell_count);
            strc.buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(strc.internal_cell_count);
            strc.buf_endrange    = std::make_unique<sycl::buffer<u32>>(strc.internal_cell_count);

            serializer.load_buf(*strc.buf_lchild_id, strc.internal_cell_count);
            serializer.load_buf(*strc.buf_rchild_id, strc.internal_cell_count);
            serializer.load_buf(*strc.buf_lchild_flag, strc.internal_cell_count);
            serializer.load_buf(*strc.buf_rchild_flag, strc.internal_cell_count);
            serializer.load_buf(*strc.buf_endrange, strc.internal_cell_count);

            return strc;
        }
    };

} // namespace shamrock::tree
