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
 * @file serialize.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "details/SerializeHelperMember.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace shamalgs {

    struct SerializeSize {
        u64 head_size    = 0;
        u64 content_size = 0;

        SerializeSize &
        operator+=(const SerializeSize &rhs) // compound assignment (does not need to be a member,
        {                                    // but often is, to modify the private members)
            /* addition of rhs to *this takes place here */
            head_size += rhs.head_size;
            content_size += rhs.content_size;
            return *this; // return the result by reference
        }

        // friends defined inside class body are inline and are hidden from non-ADL lookup
        friend SerializeSize operator+(
            SerializeSize lhs,        // passing lhs by value helps optimize chained a+b+c
            const SerializeSize &rhs) // otherwise, both parameters may be const references
        {
            lhs += rhs; // reuse compound assignment
            return lhs; // return the result by value (uses move constructor)
        }
        SerializeSize &
        operator*=(const SerializeSize &rhs) // compound assignment (does not need to be a member,
        {                                    // but often is, to modify the private members)
            /* addition of rhs to *this takes place here */
            head_size *= rhs.head_size;
            content_size *= rhs.content_size;
            return *this; // return the result by reference
        }

        // friends defined inside class body are inline and are hidden from non-ADL lookup
        friend SerializeSize operator*(
            SerializeSize lhs,        // passing lhs by value helps optimize chained a*b*c
            const SerializeSize &rhs) // otherwise, both parameters may be const references
        {
            lhs *= rhs; // reuse compound assignment
            return lhs; // return the result by value (uses move constructor)
        }

        SerializeSize &
        operator*=(const int &rhs) // compound assignment (does not need to be a member,
        {                          // but often is, to modify the private members)
            /* addition of rhs to *this takes place here */
            head_size *= rhs;
            content_size *= rhs;
            return *this; // return the result by reference
        }

        // friends defined inside class body are inline and are hidden from non-ADL lookup
        friend SerializeSize operator*(
            SerializeSize lhs, // passing lhs by value helps optimize chained a*b*c
            const int &rhs)    // otherwise, both parameters may be const references
        {
            lhs *= rhs; // reuse compound assignment
            return lhs; // return the result by value (uses move constructor)
        }

        static SerializeSize Header(u64 sz) { return {sz, 0}; }
        static SerializeSize Content(u64 sz) { return {0, sz}; }

        inline u64 get_total_size() { return head_size + content_size; }
    };

    namespace details {
        template<u64 alignment>
        inline u64 align_repr(u64 offset) {
            u64 modval = offset % alignment;
            if (modval == 0) {
                return offset;
            }
            return offset + (alignment - modval);
        }

        template<u64 alignment, class T>
        inline SerializeSize serialize_byte_size() {
            using Helper = details::SerializeHelperMember<T>;
            return SerializeSize::Header(align_repr<alignment>(Helper::szrepr));
        }

        template<u64 alignment, class T>
        inline SerializeSize serialize_byte_size(u64 len) {
            using Helper = details::SerializeHelperMember<T>;
            return SerializeSize::Content(align_repr<alignment>(len * Helper::szrepr));
        }

        template<u64 alignment>
        inline SerializeSize serialize_byte_size(std::string s) {
            return serialize_byte_size<alignment, u32>()
                   + serialize_byte_size<alignment, u32>(s.size());
        }

    } // namespace details

    class SerializeHelper {

        u64 header_size = 0;

        sham::DeviceBuffer<u8> storage;
        std::vector<u8> storage_header = {};
        u64 head_device                = 0;
        u64 head_host                  = 0;

        static constexpr u64 alignment = 8;

        template<class T>
        inline void check_head_move_device(u64 off, u64 len) {
            using Helper = details::SerializeHelperMember<T>;

            SHAM_ASSERT(off == align_repr(len * Helper::szrepr));

            if (head_device + off > storage.get_size()) {
                throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                    "the buffer is not allocated, the head_device cannot be moved\n "
                    "    storage size : {}\n"
                    "    current head_device : {}\n"
                    "    requested head_device : {}\n"
                    "    offset : {}\n"
                    "    len : {}\n"
                    "    Helper::szrepr : {}",
                    storage.get_size(),
                    head_device,
                    head_device + off,
                    off,
                    len,
                    Helper::szrepr));
            }
        }

        template<class T>
        inline void check_head_move_host(u64 off, u64 len) {
            using Helper = details::SerializeHelperMember<T>;

            SHAM_ASSERT(off == align_repr(len * Helper::szrepr));

            if (head_host + off > storage_header.size()) {
                throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                    "the buffer is not allocated, the head_host cannot be moved\n "
                    "    storage_header size : {}\n"
                    "    current head_host : {}\n"
                    "    requested head_host : {}\n"
                    "    offset : {}\n"
                    "    len : {}\n"
                    "    Helper::szrepr : {}",
                    storage_header.size(),
                    head_host,
                    head_host + off,
                    off,
                    len,
                    Helper::szrepr));
            }
        }

        inline static u64 align_repr(u64 offset) { return details::align_repr<alignment>(offset); }

        static u64 pre_head_length();

        std::shared_ptr<sham::DeviceScheduler> dev_sched;

        public:
        std::shared_ptr<sham::DeviceScheduler> &get_device_scheduler() { return dev_sched; }

        SerializeHelper(std::shared_ptr<sham::DeviceScheduler> dev_sched);

        SerializeHelper(
            std::shared_ptr<sham::DeviceScheduler> dev_sched, sham::DeviceBuffer<u8> &&storage);

        void allocate(SerializeSize szinfo);

        sham::DeviceBuffer<u8> finalize();

        template<class T>
        inline static SerializeSize serialize_byte_size() {
            return details::serialize_byte_size<alignment, T>();
        }

        template<class T>
        inline static SerializeSize serialize_byte_size(u64 len) {
            return details::serialize_byte_size<alignment, T>(len);
        }

        inline static SerializeSize serialize_byte_size(std::string s) {
            return details::serialize_byte_size<alignment>(s);
        }

        template<class T>
        inline void write(T val) {
            StackEntry stack_loc{false};

            using Helper = details::SerializeHelperMember<T>;

            u64 current_head = head_host;

            u64 offset = align_repr(Helper::szrepr);
            check_head_move_host<T>(offset, 1);

            Helper::store(&(storage_header)[current_head], val);

            head_host += offset;
        }

        template<class T>
        inline void load(T &val) {
            StackEntry stack_loc{false};

            using Helper = details::SerializeHelperMember<T>;

            u64 current_head = head_host;
            u64 offset       = align_repr(Helper::szrepr);
            check_head_move_host<T>(offset, 1);

            { // using host_acc rather than anything else since other options causes addition
              // latency

                val = Helper::load(&(storage_header)[current_head]);
            }

            head_host += offset;
        }

        inline void write(std::string s) {
            StackEntry stack_loc{false};
            write(u32(s.size()));

            sycl::buffer<char> buf(s.size());
            {
                sycl::host_accessor acc{buf, sycl::write_only, sycl::no_init};
                for (u32 i = 0; i < s.size(); i++) {
                    acc[i] = s[i];
                }
            }
            write_buf(buf, s.size());
        }

        inline void load(std::string &s) {
            StackEntry stack_loc{false};
            u32 len;
            load(len);
            s.resize(len);

            sycl::buffer<char> buf(len);
            load_buf(buf, len);
            {
                sycl::host_accessor acc{buf, sycl::read_only};
                for (u32 i = 0; i < len; i++) {
                    s[i] = acc[i];
                }
            }
        }

        template<class T>
        inline void write_buf(sycl::buffer<T> &buf, u64 len) {
            StackEntry stack_loc{false};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head_device;

            u64 offset = align_repr(len * Helper::szrepr);
            check_head_move_device<T>(offset, len);

            sham::EventList depends_list;

            auto accbufbyte = storage.get_write_access(depends_list);

            auto e = dev_sched->get_queue().submit(
                depends_list, [&, current_head](sycl::handler &cgh) {
                    sycl::accessor accbuf{buf, cgh, sycl::read_only};

                    cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                        u64 head = current_head + id.get_linear_id() * Helper::szrepr;
                        Helper::store(&accbufbyte[head], accbuf[id]);
                    });
                });

            storage.complete_event_state(e);

            head_device += offset;
        }

        template<class T>
        inline void load_buf(sycl::buffer<T> &buf, u64 len) {
            StackEntry stack_loc{false};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head_device;

            u64 offset = align_repr(len * Helper::szrepr);
            check_head_move_device<T>(offset, len);

            sham::EventList depends_list;

            auto accbufbyte = storage.get_read_access(depends_list);

            auto e = dev_sched->get_queue().submit(
                depends_list, [&, current_head](sycl::handler &cgh) {
                    sycl::accessor accbuf{buf, cgh, sycl::write_only, sycl::no_init};

                    cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                        u64 head   = current_head + id.get_linear_id() * Helper::szrepr;
                        accbuf[id] = Helper::load(&accbufbyte[head]);
                    });
                });

            storage.complete_event_state(e);

            head_device += offset;
        }

        template<class T>
        inline void write_buf(sham::DeviceBuffer<T> &buf, u64 len) {
            StackEntry stack_loc{false};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head_device;

            u64 offset = align_repr(len * Helper::szrepr);
            check_head_move_device<T>(offset, len);

            sham::EventList depends_list;
            const T *accbuf = buf.get_read_access(depends_list);
            auto accbufbyte = storage.get_write_access(depends_list);

            auto e = dev_sched->get_queue().submit(
                depends_list, [&, current_head](sycl::handler &cgh) {
                    cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                        u64 head = current_head + id.get_linear_id() * Helper::szrepr;
                        Helper::store(&accbufbyte[head], accbuf[id]);
                    });
                });

            buf.complete_event_state(e);
            storage.complete_event_state(e);

            head_device += offset;
        }

        template<class T>
        inline void load_buf(sham::DeviceBuffer<T> &buf, u64 len) {
            StackEntry stack_loc{false};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head_device;

            u64 offset = align_repr(len * Helper::szrepr);
            check_head_move_device<T>(offset, len);

            if (buf.get_size() < len) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "SerializeHelper::load_buf: (buf.get_size() < len)\n  buf.get_size()={}\n  "
                    "len={}",
                    buf.get_size(),
                    len));
            }

            sham::EventList depends_list;
            T *accbuf       = buf.get_write_access(depends_list);
            auto accbufbyte = storage.get_read_access(depends_list);

            auto e = dev_sched->get_queue().submit(
                depends_list, [&, current_head](sycl::handler &cgh) {
                    cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                        u64 head   = current_head + id.get_linear_id() * Helper::szrepr;
                        accbuf[id] = Helper::load(&accbufbyte[head]);
                    });
                });

            buf.complete_event_state(e);
            storage.complete_event_state(e);

            head_device += offset;
        }
    };

} // namespace shamalgs
