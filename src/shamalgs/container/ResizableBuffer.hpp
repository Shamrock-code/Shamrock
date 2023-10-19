// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


/**
 * @file ResizableBuffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 

#include "shambase/type_aliases.hpp"
#include "shambase/sycl_vec_aliases.hpp"
#include "shamalgs/algorithm.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/serialize.hpp"
#include "shambase/exception.hpp"

#define XMAC_LIST_ENABLED_ResizableUSMBuffer                                                       \
    X(f32)                                                                                         \
    X(f32_2)                                                                                       \
    X(f32_3)                                                                                       \
    X(f32_4)                                                                                       \
    X(f32_8)                                                                                       \
    X(f32_16)                                                                                      \
    X(f64)                                                                                         \
    X(f64_2)                                                                                       \
    X(f64_3)                                                                                       \
    X(f64_4)                                                                                       \
    X(f64_8)                                                                                       \
    X(f64_16)                                                                                      \
    X(u32)                                                                                         \
    X(u64)                                                                                         \
    X(u32_3)                                                                                       \
    X(u64_3)                                                                                       \
    X(i64_3)

namespace shamalgs {
    template<class T>
    class ResizableBuffer {

        // clang-format off
        static constexpr bool is_in_type_list =
            #define X(args) std::is_same<T, args>::value ||
            XMAC_LIST_ENABLED_ResizableUSMBuffer false
            #undef X
            ;

        static_assert(
            is_in_type_list,
            "PatchDataField must be one of those types : "
            #define X(args) #args " "
            XMAC_LIST_ENABLED_ResizableUSMBuffer
            #undef X
        );
        // clang-format on

        std::unique_ptr<sycl::buffer<T>> buf;

        u32 capacity;
        u32 val_cnt; // nvar*obj_cnt

        constexpr static u32 min_capa  = 100;
        constexpr static f32 safe_fact = 1.25;

        void alloc();
        void free();
        void change_capacity(u32 new_capa);

        public:
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // memory manipulation
        ////////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the reference to the internal buffer
         *
         * @return const std::unique_ptr<sycl::buffer<T>>&
         */
        inline const std::unique_ptr<sycl::buffer<T>> &get_buf() const { return buf; }

        /**
         * @brief Get a editable reference to the internal buffer
         *
         * @return const std::unique_ptr<sycl::buffer<T>>&
         */
        inline std::unique_ptr<sycl::buffer<T>> &get_buf_priviledge() { return buf; }

        /**
         * @brief get the number of held values in the buffer
         *
         * @return const u32&
         */
        [[nodiscard]] inline const u32 &size() const { return val_cnt; }

        /**
         * @brief get the memsize of the buffer
         *
         * @return u64
         */
        [[nodiscard]] inline u64 memsize() const { return val_cnt * sizeof(T); }

        /**
         * @brief resize the buffer size
         *
         * @param new_size
         */
        void resize(u32 new_size);

        /**
         * @brief reserve slots for the buffer the buffer size
         *
         * @param add_size
         */
        void reserve(u32 add_size);

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // value manipulation
        ////////////////////////////////////////////////////////////////////////////////////////////////

        void overwrite(ResizableBuffer<T> &f2, u32 cnt);

        void override(sycl::buffer<T> &data, u32 cnt);

        /**
         * @brief override the field data with the value specified in `val`
         * \todo missing test
         * @param val
         */
        void override(const T val);

        void index_remap_resize(sycl::buffer<u32> &index_map, u32 len, u32 nvar = 1);

        [[nodiscard]] bool check_buf_match(const ResizableBuffer<T> &f2) const;

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // constructors & destructors
        ////////////////////////////////////////////////////////////////////////////////////////////////

        explicit ResizableBuffer(u32 cnt = 0) : val_cnt(cnt), capacity(cnt) {
            if (capacity != 0) {
                alloc();
            }
        }

        inline ResizableBuffer(sycl::buffer<T> &&moved_buf, u32 val_cnt)
            : buf(std::make_unique<sycl::buffer<T>>(std::move(moved_buf))), val_cnt(val_cnt),
              capacity(moved_buf.size()) {}

        ResizableBuffer(const ResizableBuffer &other)
            : val_cnt(other.val_cnt), capacity(other.capacity) {
            if (capacity != 0) {
                alloc();
                // copydata(other._data,_data, capacity);
                shamalgs::memory::copybuf_discard(*other.buf, *buf, capacity);
            }
        }

        ResizableBuffer(ResizableBuffer &&other) noexcept
            : buf(std::move(other.buf)), val_cnt(std::move(other.val_cnt)),
              capacity(std::move(other.capacity)) {} // move constructor

        ResizableBuffer &operator=(ResizableBuffer &&other) noexcept {
            buf      = std::move(other.buf);
            val_cnt  = std::move(other.val_cnt);
            capacity = std::move(other.capacity);

            return *this;
        } // move assignment

        /**
         * @brief serialize the content of the buffer
         *  Note : no size information will be written
         * @param serializer
         */
        void serialize_buf(shamalgs::SerializeHelper &serializer);

        /**
         * @brief inverse operation of @serialize_buf
         *
         * @param serializer
         * @param val_cnt
         * @return ResizableBuffer
         */
        static ResizableBuffer deserialize_buf(shamalgs::SerializeHelper &serializer, u32 val_cnt);

        u64 serialize_buf_byte_size();

        static ResizableBuffer mock_buffer(u64 seed, u32 val_cnt, T min_bound, T max_bound);

        ResizableBuffer &operator=(const ResizableBuffer &other) // copy assignment
            = delete;

        inline ~ResizableBuffer() { free(); }
    };
} // namespace shamalgs