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
 * @file USMBufferInterop.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamcomm/logs.hpp"

namespace sham {
    /**
     * @brief perform a copy from a buffer to a USM pointer
     *
     * @tparam T
     * @param queue
     * @param src
     * @param dest
     * @param count
     * @return std::vector<sycl::event>
     */
    template<class T>
    inline std::vector<sycl::event>
    usmbuffer_memcpy(sycl::queue &queue, sycl::buffer<T> &src, T *dest, u64 count) {

        u64 offset                  = 0;
        u64 remains                 = count;
        constexpr u64 max_step_size = i32_max / 2;

        std::vector<sycl::event> ev_list{};

        while (offset < count) {
            u64 stepsize = sham::min(remains, max_step_size);

            ev_list.push_back(queue.submit([&, offset](sycl::handler &cgh) {
                sycl::accessor acc{src, cgh, sycl::read_only};
                shambase::parallel_for(cgh, stepsize, "memcpy kernel", [=](u32 gid) {
                    dest[gid + offset] = acc[gid + offset];
                });
            }));

            offset += stepsize;
            remains -= stepsize;
        };

        return ev_list;
    }

    /**
     * @brief perform a copy from a USM pointer to a buffer
     *
     * @tparam T
     * @param queue
     * @param src
     * @param dest
     * @param count
     * @return std::vector<sycl::event>
     */
    template<class T>
    inline std::vector<sycl::event>
    usmbuffer_memcpy(sycl::queue &queue, const T *src, sycl::buffer<T> &dest, u64 count) {

        u64 offset                  = 0;
        u64 remains                 = count;
        constexpr u64 max_step_size = i32_max / 2;

        std::vector<sycl::event> ev_list{};

        while (offset < count) {
            u64 stepsize = sham::min(remains, max_step_size);

            ev_list.push_back(queue.submit([&, offset](sycl::handler &cgh) {
                sycl::accessor acc{dest, cgh, sycl::write_only};
                shambase::parallel_for(cgh, stepsize, "memcpy kernel", [=](u32 gid) {
                    acc[gid + offset] = src[gid + offset];
                });
            }));

            offset += stepsize;
            remains -= stepsize;
        };

        return ev_list;
    }

    /**
     * @brief perform a copy from a USM pointer to a buffer (and assume discard write for the
     * buffer)
     *
     * @tparam T
     * @param queue
     * @param src
     * @param dest
     * @param count
     * @return std::vector<sycl::event>
     */
    template<class T>
    inline std::vector<sycl::event>
    usmbuffer_memcpy_discard(sycl::queue &queue, const T *src, sycl::buffer<T> &dest, u64 count) {
        u64 offset                  = 0;
        u64 remains                 = count;
        constexpr u64 max_step_size = i32_max / 2;

        std::vector<sycl::event> ev_list{};

        while (offset < count) {
            u64 stepsize = sham::min(remains, max_step_size);

            ev_list.push_back(queue.submit([&, offset](sycl::handler &cgh) {
                sycl::accessor acc{dest, cgh, sycl::write_only, sycl::no_init};
                shambase::parallel_for(cgh, stepsize, "memcpy kernel", [=](u32 gid) {
                    acc[gid + offset] = src[gid + offset];
                });
            }));

            offset += stepsize;
            remains -= stepsize;
        };

        return ev_list;
    }

} // namespace sham
