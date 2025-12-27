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
 * @file memory.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::memory {

    /**
     * @brief extract a value of a buffer
     *
     * @tparam T the type of the buffer & value
     * @param q the queue to use
     * @param buf the buffer to extract from
     * @param idx the index of the value that will be extracted
     * @return T the extracted value
     */
    template<class T>
    T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

    /**
     * @brief Extract a value from a DeviceBuffer
     *
     * @tparam T Type of buffer elements
     * @param q Device queue to submit operation
     * @param buf Device buffer to extract from
     * @param idx Index of element to extract
     * @return T the extracted value
     */
    template<class T>
    T extract_element(sham::DeviceQueue &q, sham::DeviceBuffer<T> &buf, u32 idx) {
        T val;

        sham::EventList depends_list;
        auto acc = buf.get_read_access(depends_list);

        T *dest = &val;

        q.submit(depends_list, [&, idx](sycl::handler &cgh) {
             cgh.copy(acc + idx, dest, 1);
         }).wait_and_throw();
        ;

        buf.complete_event_state(sycl::event{});

        return val;
    }

    template<class T>
    void set_element(
        sycl::queue &q, sycl::buffer<T> &buf, u32 idx, T val, bool discard_write = false) {

        if (discard_write) {
            q.submit([&, idx, val](sycl::handler &cgh) {
                sycl::accessor acc{buf, cgh, sycl::write_only, sycl::no_init};
                cgh.single_task([=]() {
                    acc[idx] = val;
                });
            });
        } else {
            q.submit([&, idx, val](sycl::handler &cgh) {
                sycl::accessor acc{buf, cgh, sycl::write_only};
                cgh.single_task([=]() {
                    acc[idx] = val;
                });
            });
        }
    }

    /**
     * @brief Convert a `std::vector` to a `sycl::buffer`
     *
     * @tparam T
     * @param buf
     * @return sycl::buffer<T>
     */
    template<class T>
    sycl::buffer<T> vec_to_buf(const std::vector<T> &buf);

    /**
     * @brief Convert a `sycl::buffer` to a `std::vector`
     *
     * @tparam T
     * @param buf
     * @param len
     * @return std::vector<T>
     */
    template<class T>
    std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len);

    /**
     * @brief enqueue a do nothing kernel to force the buffer to move
     *
     * @tparam T
     * @param q
     * @param buf
     */
    template<class T>
    inline void move_buffer_on_queue(sycl::queue &q, sycl::buffer<T> &buf) {
        sycl::buffer<T> tmp(1);
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor a{buf, cgh, sycl::read_write};
            sycl::accessor b{tmp, cgh, sycl::write_only, sycl::no_init};

            cgh.single_task([=]() {
                b[0] = a[0];
            });
        });
    }

    /**
     * @brief Fill a buffer with a given value
     *
     * @tparam T
     * @param q
     * @param buf
     * @param value
     */
    template<class T>
    inline void buf_fill(sycl::queue &q, sycl::buffer<T> &buf, T value) {
        StackEntry stack_loc{};
        q.submit([&, value](sycl::handler &cgh) {
            sycl::accessor acc{buf, cgh, sycl::write_only};
            shambase::parallel_for(cgh, buf.size(), "buf_fill", [=](u64 id_a) {
                acc[id_a] = value;
            });
        });
    }

    /**
     * @brief Fill a buffer with a given value (sycl::no_init mode)
     *
     * @tparam T
     * @param q
     * @param buf
     * @param value
     */
    template<class T>
    inline void buf_fill_discard(sycl::queue &q, sycl::buffer<T> &buf, T value) {
        StackEntry stack_loc{};
        q.submit([&, value](sycl::handler &cgh) {
            sycl::accessor acc{buf, cgh, sycl::write_only, sycl::no_init};

            shambase::parallel_for(cgh, buf.size(), "buff_fill_discard", [=](u64 id_a) {
                acc[id_a] = value;
            });
        });
    }

    /**
     * @brief Print the content of a `sycl::buffer`
     *
     * @tparam T
     * @tparam Tformat
     * @param buf
     * @param len
     * @param column_count
     * @param fmt
     */
    template<class T, typename... Tformat>
    inline void print_buf(
        sycl::buffer<T> &buf, u32 len, u32 column_count, fmt::format_string<Tformat...> fmt) {

        sycl::host_accessor acc{buf, sycl::read_only};

        std::string accum;

        for (u32 i = 0; i < len; i++) {

            if (i % column_count == 0) {
                if (i == 0) {
                    accum += shambase::format("{:8} : ", i);
                } else {
                    accum += shambase::format("\n{:8} : ", i);
                }
            }

            accum += shambase::format(fmt, acc[i]);
        }

        logger::raw_ln(accum);
    }

    template<class T>
    void copybuf_discard(sycl::queue &q, sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt) {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor src{source, cgh, sycl::read_only};
            sycl::accessor dst{dest, cgh, sycl::write_only, sycl::no_init};

            shambase::parallel_for(cgh, cnt, "copybuf_discard", [=](u64 i) {
                dst[i] = src[i];
            });
        });
    }

    template<class T>
    void copybuf(sycl::queue &q, sycl::buffer<T> &source, sycl::buffer<T> &dest, u32 cnt) {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor src{source, cgh, sycl::read_only};
            sycl::accessor dst{dest, cgh, sycl::write_only};

            shambase::parallel_for(cgh, cnt, "copybuf", [=](u64 i) {
                dst[i] = src[i];
            });
        });
    }

    template<class T>
    void add_with_factor_to(
        sycl::queue &q, sycl::buffer<T> &buf, T factor, sycl::buffer<T> &op, u32 cnt) {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc{buf, cgh, sycl::read_write};
            sycl::accessor dd{op, cgh, sycl::read_only};

            T fac = factor;

            shambase::parallel_for(cgh, cnt, "add_with_factor_to", [=](u64 i) {
                acc[i] += fac * dd[i];
            });
        });
    }

    template<class T>
    void write_with_offset_into(
        sycl::queue &q,
        sycl::buffer<T> &buf_ctn,
        sycl::buffer<T> &buf_in,
        u32 offset,
        u32 element_count) {
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor source{buf_in, cgh, sycl::read_only};
            sycl::accessor dest{buf_ctn, cgh, sycl::write_only, sycl::no_init};
            u32 off = offset;
            cgh.parallel_for(sycl::range{element_count}, [=](sycl::item<1> item) {
                dest[item.get_id(0) + off] = source[item];
            });
        });
    }

    /**
     * @brief Write elements from input buffer to container buffer at an offset (DeviceBuffer
     * variant)
     *
     * @tparam T Type of buffer elements
     * @param q Device queue to submit operation
     * @param buf_ctn Container buffer (destination)
     * @param buf_in Input buffer (source)
     * @param offset Starting position in container
     * @param element_count Number of elements to write
     */
    template<class T>
    void write_with_offset_into(
        sham::DeviceQueue &q,
        sham::DeviceBuffer<T> &buf_ctn,
        sham::DeviceBuffer<T> &buf_in,
        u32 offset,
        u32 element_count) {

        sham::EventList depends_list;
        auto source = buf_in.get_read_access(depends_list);
        auto dest   = buf_ctn.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            u32 off = offset;
            cgh.parallel_for(sycl::range{element_count}, [=](sycl::item<1> item) {
                dest[item.get_id(0) + off] = source[item];
            });
        });

        buf_in.complete_event_state(e);
        buf_ctn.complete_event_state(e);
    }

    /**
     * @brief Write elements from DeviceBuffer to sycl::buffer at an offset
     *
     * @tparam T
     * @param q
     * @param buf_ctn Container buffer (destination)
     * @param buf_in Input DeviceBuffer (source)
     * @param offset Starting position in container
     * @param element_count Number of elements to write
     */
    template<class T>
    void write_with_offset_into(
        sham::DeviceQueue &q,
        sycl::buffer<T> &buf_ctn,
        sham::DeviceBuffer<T> &buf_in,
        u32 offset,
        u32 element_count) {

        sham::EventList depends_list;
        auto source = buf_in.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            sycl::accessor dest{buf_ctn, cgh, sycl::write_only, sycl::no_init};
            u32 off = offset;
            cgh.parallel_for(sycl::range{element_count}, [=](sycl::item<1> item) {
                dest[item.get_id(0) + off] = source[item];
            });
        });

        buf_in.complete_event_state(e);
    }

    template<class T>
    void write_with_offset_into(
        sycl::queue &q, sycl::buffer<T> &buf_ctn, T val, u32 offset, u32 element_count) {
        q.submit([&, val](sycl::handler &cgh) {
            sycl::accessor dest{buf_ctn, cgh, sycl::write_only, sycl::no_init};
            u32 off = offset;
            cgh.parallel_for(sycl::range{element_count}, [=](sycl::item<1> item) {
                dest[item.get_id(0) + off] = val;
            });
        });
    }

    template<class T>
    std::unique_ptr<sycl::buffer<T>> duplicate(
        sycl::queue &q, const std::unique_ptr<sycl::buffer<T>> &buf_in) {
        if (buf_in) {
            auto buf = std::make_unique<sycl::buffer<T>>(buf_in->size());
            copybuf_discard(q, *buf_in, *buf, buf_in->size());
            return std::move(buf);
        }
        return {};
    }

    template<class T>
    sycl::buffer<T> vector_to_buf(sycl::queue &q, std::vector<T> &&vec) {

        u32 cnt = vec.size();
        sycl::buffer<T> ret(cnt);

        sycl::buffer<T> alias(vec.data(), cnt);

        shamalgs::memory::copybuf_discard(q, alias, ret, cnt);

// HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer
// doesn't wait for the end of the queue resulting in out of bound access
#ifdef SYCL_COMP_ACPP
        q.wait();
#endif

        return std::move(ret);
    }

    /**
     * @brief Convert std::vector to sycl::buffer with data copy (lvalue reference version)
     *
     * @tparam T Type of vector elements
     * @param q SYCL queue to submit operation
     * @param vec Vector to convert
     * @return sycl::buffer<T>
     */
    template<class T>
    sycl::buffer<T> vector_to_buf(sycl::queue &q, std::vector<T> &vec) {

        u32 cnt = vec.size();
        sycl::buffer<T> ret(cnt);

        sycl::buffer<T> alias(vec.data(), cnt);

        shamalgs::memory::copybuf_discard(q, alias, ret, cnt);

// HIPSYCL segfault otherwise because looks like the destructor of the sycl buffer
// doesn't wait for the end of the queue resulting in out of bound access
#ifdef SYCL_COMP_ACPP
        q.wait();
#endif

        return std::move(ret);
    }

} // namespace shamalgs::memory
