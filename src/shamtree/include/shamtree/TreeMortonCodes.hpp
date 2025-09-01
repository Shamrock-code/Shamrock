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
 * @file TreeMortonCodes.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/details/memory/memory.hpp"
#include "shamalgs/serialize.hpp"
#include "shammath/CoordRange.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

namespace shamrock::tree {

    template<class u_morton>
    class TreeMortonCodes {
        public:
        u32 obj_cnt;

        std::unique_ptr<sycl::buffer<u_morton>> buf_morton;
        std::unique_ptr<sycl::buffer<u32>> buf_particle_index_map;

        template<class T>
        void build(
            sycl::queue &queue,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sycl::buffer<T> &pos_buf);

        template<class T>
        void build(
            sham::DeviceScheduler_ptr dev_sched,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sham::DeviceBuffer<T> &pos_buf);

        template<class T>
        static std::unique_ptr<sycl::buffer<u_morton>> build_raw(
            sycl::queue &queue,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sycl::buffer<T> &pos_buf);

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            sum += sizeof(obj_cnt);

            add_ptr(buf_morton);
            add_ptr(buf_particle_index_map);

            return sum;
        }

        inline TreeMortonCodes() = default;

        inline TreeMortonCodes(const TreeMortonCodes &other)
            : obj_cnt(other.obj_cnt),
              buf_morton(
                  shamalgs::memory::duplicate(
                      shamsys::instance::get_compute_queue(), other.buf_morton)),
              buf_particle_index_map(
                  shamalgs::memory::duplicate(
                      shamsys::instance::get_compute_queue(), other.buf_particle_index_map)) {}

        inline TreeMortonCodes &operator=(TreeMortonCodes &&other) noexcept {
            obj_cnt                = std::move(other.obj_cnt);
            buf_morton             = std::move(other.buf_morton);
            buf_particle_index_map = std::move(other.buf_particle_index_map);

            return *this;
        } // move assignment

        bool operator==(const TreeMortonCodes &rhs) const;

        /**
         * @brief serialize a TreeMortonCodes object
         *
         * @param serializer
         */
        inline void serialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};

            serializer.write(obj_cnt);
            if (!buf_morton) {
                throw shambase::make_except_with_loc<std::runtime_error>("missing buffer");
            }
            // serializer.write(buf_morton->size());
            serializer.write_buf(*buf_morton, obj_cnt);
            if (!buf_particle_index_map) {
                throw shambase::make_except_with_loc<std::runtime_error>("missing buffer");
            }
            serializer.write_buf(*buf_particle_index_map, obj_cnt);
        }

        /**
         * @brief deserialize a TreeMortonCodes object
         * Note : here since the initial buffer is a pow of 2
         * with trailling terms for the bitonic sort, when
         * deserializing we are not loading the last values
         * the buffer size is obj_cnt here
         *
         * @param serializer
         * @return TreeMortonCodes
         */
        inline static TreeMortonCodes deserialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};
            TreeMortonCodes ret;
            serializer.load(ret.obj_cnt);

            // u32 morton_len;
            // serializer.load(morton_len);
            ret.buf_morton             = std::make_unique<sycl::buffer<u_morton>>(ret.obj_cnt);
            ret.buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(ret.obj_cnt);

            serializer.load_buf(*ret.buf_morton, ret.obj_cnt);
            serializer.load_buf(*ret.buf_particle_index_map, ret.obj_cnt);

            return ret;
        }

        /**
         * @brief give the size of the serialized object
         *
         * @return u64
         */
        inline shamalgs::SerializeSize serialize_byte_size() {
            using H = shamalgs::SerializeHelper;
            return H::serialize_byte_size<u32>() + H::serialize_byte_size<u32>(obj_cnt)
                   + H::serialize_byte_size<u_morton>(obj_cnt);
        }
    };

} // namespace shamrock::tree
