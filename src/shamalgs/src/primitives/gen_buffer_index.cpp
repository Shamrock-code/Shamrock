// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file gen_buffer_index.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements functions to generate and fill buffers with sequential indices.
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives {

    void fill_buffer_index(sham::DeviceBuffer<u32> &buf, u32 len) {
        if (buf.get_size() < len) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "buf.get_size() < len\n  buf.get_size() = {},\n  len = {}", buf.get_size(), len));
        }

        sham::kernel_call(
            buf.get_queue(), sham::MultiRef{}, sham::MultiRef{buf}, len, [](u32 i, u32 *idx) {
                idx[i] = i;
            });
    }

    sham::DeviceBuffer<u32> gen_buffer_index(sham::DeviceScheduler_ptr sched, u32 len) {
        sham::DeviceBuffer<u32> ret(len, sched);

        fill_buffer_index(ret, len);

        return ret;
    }
} // namespace shamalgs::primitives
