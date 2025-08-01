// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file algorithm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::algorithm {

    sycl::buffer<u32> gen_buffer_index(sycl::queue &q, u32 len) {
        return gen_buffer_device(q, len, [](u32 i) -> u32 {
            return i;
        });
    }

    void
    fill_buffer_index_usm(sham::DeviceScheduler_ptr sched, u32 len, sham::DeviceBuffer<u32> &buf) {
        buf.resize(len);

        sham::kernel_call(
            sched->get_queue(), sham::MultiRef{}, sham::MultiRef{buf}, len, [](u32 i, u32 *idx) {
                idx[i] = i;
            });
    }

    sham::DeviceBuffer<u32> gen_buffer_index_usm(sham::DeviceScheduler_ptr sched, u32 len) {
        sham::DeviceBuffer<u32> ret(len, sched);

        fill_buffer_index_usm(sched, len, ret);

        return ret;
    }

    template<class T>
    sycl::buffer<T>
    index_remap(sycl::queue &q, sycl::buffer<T> &buf, sycl::buffer<u32> &index_map, u32 len) {

        sycl::buffer<T> ret(len);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor in{buf, cgh, sycl::read_only};
            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor permut{index_map, cgh, sycl::read_only};

            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                out[item] = in[permut[item]];
            });
        });

        return std::move(ret);
    }

    template<class T>
    sycl::buffer<T> index_remap_nvar(
        sycl::queue &q, sycl::buffer<T> &buf, sycl::buffer<u32> &index_map, u32 len, u32 nvar) {

        sycl::buffer<T> ret(len * nvar);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor in{buf, cgh, sycl::read_only};
            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor permut{index_map, cgh, sycl::read_only};

            u32 nvar_loc = nvar;

            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                u32 in_id  = permut[item] * nvar_loc;
                u32 out_id = item.get_linear_id() * nvar_loc;

                for (u32 a = 0; a < nvar_loc; a++) {
                    out[out_id + a] = in[in_id + a];
                }
            });
        });

        return std::move(ret);
    }

    template<class T>
    void index_remap(
        const sham::DeviceScheduler_ptr &sched_ptr,
        sham::DeviceBuffer<T> &source,
        sham::DeviceBuffer<T> &dest,
        sham::DeviceBuffer<u32> &index_map,
        u32 len) {

        sham::DeviceQueue &q = shambase::get_check_ref(sched_ptr).get_queue();

        sham::EventList el;

        const T *in       = source.get_read_access(el);
        T *out            = dest.get_write_access(el);
        const u32 *permut = index_map.get_read_access(el);

        auto e = q.submit(el, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                out[item] = in[permut[item]];
            });
        });

        source.complete_event_state(e);
        dest.complete_event_state(e);
        index_map.complete_event_state(e);
    }

    template<class T>
    void index_remap_nvar(
        const sham::DeviceScheduler_ptr &sched_ptr,
        sham::DeviceBuffer<T> &source,
        sham::DeviceBuffer<T> &dest,
        sham::DeviceBuffer<u32> &index_map,
        u32 len,
        u32 nvar) {

        sham::DeviceQueue &q = shambase::get_check_ref(sched_ptr).get_queue();

        sham::EventList el;

        const T *in       = source.get_read_access(el);
        T *out            = dest.get_write_access(el);
        const u32 *permut = index_map.get_read_access(el);

        auto e = q.submit(el, [&](sycl::handler &cgh) {
            u32 nvar_loc = nvar;

            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                u32 in_id  = permut[item] * nvar_loc;
                u32 out_id = item.get_linear_id() * nvar_loc;

                for (u32 a = 0; a < nvar_loc; a++) {
                    out[out_id + a] = in[in_id + a];
                }
            });
        });

        source.complete_event_state(e);
        dest.complete_event_state(e);
        index_map.complete_event_state(e);
    }

#define XMAC_TYPES                                                                                 \
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
    X(i64_3)                                                                                       \
    X(i64)

#define X(_arg_)                                                                                   \
    template sycl::buffer<_arg_> index_remap(                                                      \
        sycl::queue &q, sycl::buffer<_arg_> &buf, sycl::buffer<u32> &index_map, u32 len);          \
                                                                                                   \
    template sycl::buffer<_arg_> index_remap_nvar(                                                 \
        sycl::queue &q,                                                                            \
        sycl::buffer<_arg_> &buf,                                                                  \
        sycl::buffer<u32> &index_map,                                                              \
        u32 len,                                                                                   \
        u32 nvar);                                                                                 \
                                                                                                   \
    template void index_remap(                                                                     \
        const sham::DeviceScheduler_ptr &sched,                                                    \
        sham::DeviceBuffer<_arg_> &source,                                                         \
        sham::DeviceBuffer<_arg_> &dest,                                                           \
        sham::DeviceBuffer<u32> &index_map,                                                        \
        u32 len);                                                                                  \
                                                                                                   \
    template void index_remap_nvar(                                                                \
        const sham::DeviceScheduler_ptr &sched,                                                    \
        sham::DeviceBuffer<_arg_> &source,                                                         \
        sham::DeviceBuffer<_arg_> &dest,                                                           \
        sham::DeviceBuffer<u32> &index_map,                                                        \
        u32 len,                                                                                   \
        u32 nvar);

    XMAC_TYPES

#undef X

} // namespace shamalgs::algorithm
