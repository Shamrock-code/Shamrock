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
 * @file numericFallback.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"

namespace shamalgs::numeric::details {

    template<class T>
    sycl::buffer<T> exclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> inclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    void exclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    template<class T>
    void inclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    std::tuple<std::optional<sycl::buffer<u32>>, u32>
    stream_compact_fallback(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

} // namespace shamalgs::numeric::details
