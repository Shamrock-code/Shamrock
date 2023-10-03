// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include <shambackends/backends/sycl/sycl.hpp>
#include "shambase/type_aliases.hpp"
#include "shambase/sycl_vec_aliases.hpp"
#include "shamsys/legacy/log.hpp"
#include <cstddef>
#include <stdexcept>
#include <utility>

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

    enum BufferType { Host, Device, Shared };

    template<class T>
    class ResizableUSMBuffer {

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

        sycl::queue &q;

        T *usm_ptr = nullptr;

        u32 capacity  = 0;
        u32 val_count = 0;

        constexpr static u32 min_capa  = 100;
        constexpr static f32 safe_fact = 1.25;

        BufferType type = Host;

        public:
        inline ResizableUSMBuffer(sycl::queue &q, BufferType type) : q(q), type(type){};

        ResizableUSMBuffer(const ResizableUSMBuffer &other)
            : val_count(other.val_count), capacity(other.capacity), q(other.q) {
            if (capacity != 0) {
                alloc();

                q.memcpy(usm_ptr, other.usm_ptr, sizeof(T) * val_count).wait();
            }
        }// copy constructor

        ResizableUSMBuffer(ResizableUSMBuffer &&other) noexcept
            : usm_ptr(other.release_usm_ptr()), val_count(std::move(other.val_count)),
              capacity(std::move(other.capacity)), q(other.q) {} // move constructor

        ResizableUSMBuffer &operator=(ResizableUSMBuffer &&other) noexcept {
            q         = other.q;
            usm_ptr   = std::move(other.release_usm_ptr());
            val_count = std::move(other.val_count);
            capacity  = std::move(other.capacity);

            return *this;
        } // move assignment

        ResizableUSMBuffer &operator=(const ResizableUSMBuffer &other) // copy assignment
            = delete;

        void alloc();

        void free();

        ~ResizableUSMBuffer();

        void change_capacity(u32 new_capa);

        void reserve(u32 add_size);

        void resize(u32 new_size);

        bool is_empty() { return usm_ptr == nullptr; }

        [[nodiscard]] BufferType get_buf_type() const { return type; }

        inline T *release_usm_ptr() { return std::exchange(usm_ptr, nullptr); }

        inline T const *get_usm_ptr_read_only() {
            if (is_empty()) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "the usm buffer is not allocated");
            }
            return usm_ptr;
        }

        inline T const *get_usm_ptr() {
            if (is_empty()) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "the usm buffer is not allocated");
            }
            return usm_ptr;
        }

        void change_buf_type(BufferType new_type);
    };

} // namespace shamalgs