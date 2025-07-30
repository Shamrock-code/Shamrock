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
 * @file TreeStructure.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "kernels/karras_alg.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/reduction.hpp"
#include "shamalgs/serialize.hpp"
#include "shamsys/NodeInstance.hpp"

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

        inline void
        build(sycl::queue &queue, u32 _internal_cell_count, sycl::buffer<u_morton> &morton_buf) {

            if (!(_internal_cell_count < morton_buf.size())) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "morton buf must be at least with size() greater than internal_cell_count");
            }

            internal_cell_count = _internal_cell_count;

            buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_endrange    = std::make_unique<sycl::buffer<u32>>(internal_cell_count);

            sycl_karras_alg(
                queue,
                internal_cell_count,
                morton_buf,
                *buf_lchild_id,
                *buf_rchild_id,
                *buf_lchild_flag,
                *buf_rchild_flag,
                *buf_endrange);

            one_cell_mode = false;
        }

        inline void build_one_cell_mode() {
            internal_cell_count = 1;
            buf_lchild_id       = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_rchild_id       = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_lchild_flag     = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_rchild_flag     = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_endrange        = std::make_unique<sycl::buffer<u32>>(internal_cell_count);

            {
                sycl::host_accessor rchild_id{*buf_rchild_id, sycl::write_only, sycl::no_init};
                sycl::host_accessor lchild_id{*buf_lchild_id, sycl::write_only, sycl::no_init};
                sycl::host_accessor rchild_flag{*buf_rchild_flag, sycl::write_only, sycl::no_init};
                sycl::host_accessor lchild_flag{*buf_lchild_flag, sycl::write_only, sycl::no_init};
                sycl::host_accessor endrange{*buf_endrange, sycl::write_only, sycl::no_init};

                lchild_id[0]   = 0;
                rchild_id[0]   = 1;
                lchild_flag[0] = 1;
                rchild_flag[0] = 1;

                endrange[0] = 1;
            }
            one_cell_mode = true;
        }

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

        inline friend bool operator==(const TreeStructure &t1, const TreeStructure &t2) {
            bool cmp = true;

            cmp = cmp && (t1.internal_cell_count == t2.internal_cell_count);

            cmp = cmp
                  && shamalgs::reduction::equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_lchild_id,
                      *t2.buf_lchild_id,
                      t1.internal_cell_count);
            cmp = cmp
                  && shamalgs::reduction::equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_rchild_id,
                      *t2.buf_rchild_id,
                      t1.internal_cell_count);
            cmp = cmp
                  && shamalgs::reduction::equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_lchild_flag,
                      *t2.buf_lchild_flag,
                      t1.internal_cell_count);
            cmp = cmp
                  && shamalgs::reduction::equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_rchild_flag,
                      *t2.buf_rchild_flag,
                      t1.internal_cell_count);
            cmp = cmp
                  && shamalgs::reduction::equals(
                      shamsys::instance::get_compute_queue(),
                      *t1.buf_endrange,
                      *t2.buf_endrange,
                      t1.internal_cell_count);
            cmp = cmp && (t1.one_cell_mode == t2.one_cell_mode);

            return cmp;
        }

        inline TreeStructure() = default;

        inline TreeStructure(const TreeStructure &other)
            : internal_cell_count(other.internal_cell_count), one_cell_mode(other.one_cell_mode),
              buf_lchild_id(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(), other.buf_lchild_id)), // size = internal
              buf_rchild_id(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(), other.buf_rchild_id)), // size = internal
              buf_lchild_flag(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_lchild_flag)), // size = internal
              buf_rchild_flag(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(),
                  other.buf_rchild_flag)), // size = internal
              buf_endrange(shamalgs::memory::duplicate(
                  shamsys::instance::get_compute_queue(), other.buf_endrange)) // size = internal
        {}

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
