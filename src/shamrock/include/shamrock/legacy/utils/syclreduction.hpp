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
 * @file syclreduction.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"
#include <memory>
#include <stdexcept>

namespace syclalg {

    // TODO to optimize
    template<class T>
    [[deprecated("please use the shamalgs library instead")]]
    inline T get_max(sycl::queue &queue, const std::unique_ptr<sycl::buffer<T>> &buf, u32 len) {

        T accum;

        if (buf) {
            accum = buf->get_host_access()[0];

            {
                sycl::host_accessor acc{*buf, sycl::read_only};

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < len; i++) {
                    accum = sycl::max(accum, acc[i]);
                }
            }
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "syclalg::get_max : input buffer not allocated");
        }

        return accum;
    }

    template<class T>
    [[deprecated("please use the shamalgs library instead")]]
    inline T get_max(sham::DeviceBuffer<T> &buf, u32 len) {

        T accum;

        if (!buf.is_empty()) {
            auto vec   = buf.copy_to_stdvec();
            auto accum = vec[0];

            {

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < len; i++) {
                    accum = sycl::max(accum, vec[i]);
                }
            }
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "syclalg::get_max : input buffer not allocated");
        }

        return accum;
    }

    // TODO to optimize
    template<class T>
    [[deprecated("please use the shamalgs library instead")]]
    inline T get_min(sycl::queue &queue, const std::unique_ptr<sycl::buffer<T>> &buf, u32 len) {

        T accum;

        if (buf) {

            accum = buf->get_host_access()[0];

            {
                sycl::host_accessor acc{*buf, sycl::read_only};

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < len; i++) {
                    accum = sycl::min(accum, acc[i]);
                }
            }

        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "syclalg::get_min : input buffer not allocated");
        }

        return accum;
    }

    template<class T>
    [[deprecated("please use the shamalgs library instead")]]
    inline T get_min(sham::DeviceBuffer<T> &buf, u32 len) {

        T accum;

        if (!buf.is_empty()) {
            auto vec   = buf.copy_to_stdvec();
            auto accum = vec[0];

            {

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < len; i++) {
                    accum = sycl::min(accum, vec[i]);
                }
            }
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "syclalg::get_min : input buffer not allocated");
        }

        return accum;
    }

    template<class T>
    [[deprecated("please use the shamalgs library instead")]]
    inline std::tuple<T, T> get_min_max(
        sycl::queue &queue, const std::unique_ptr<sycl::buffer<T>> &buf, u32 len) {

        T accum_min, accum_max;

        if (buf) {

            accum_min = buf->get_host_access()[0];
            accum_max = buf->get_host_access()[0];

            {
                sycl::host_accessor acc{*buf, sycl::read_only};

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < len; i++) {
                    accum_min = sycl::min(accum_min, acc[i]);
                    accum_max = sycl::max(accum_max, acc[i]);
                }
            }

        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "syclalg::get_max : input buffer not allocated");
        }

        return {accum_min, accum_max};
    }

    template<class T>
    [[deprecated("please use the shamalgs library instead")]]
    inline std::tuple<T, T> get_min_max(sham::DeviceBuffer<T> &buf, u32 len) {

        T accum_min, accum_max;

        if (!buf.is_empty()) {
            auto vec = buf.copy_to_stdvec();

            auto accum_min = vec[0];
            auto accum_max = vec[0];

            {

                // queue.submit([&](sycl::handler &cgh) {
                //     auto acc = buf->get_access<sycl::access::mode::read>(cgh);

                //     cgh.parallel_for(sycl::range(buf->size()), [=](sycl::item<1> item) {
                //         u32 i = (u32)item.get_id(0);

                //     });
                // });

                for (u32 i = 0; i < len; i++) {
                    accum_min = sycl::min(accum_min, vec[i]);
                    accum_max = sycl::max(accum_max, vec[i]);
                }
            }

        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "syclalg::get_max : input buffer not allocated");
        }

        return {accum_min, accum_max};
    }

} // namespace syclalg
