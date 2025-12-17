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
 * @file append_subset_to.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Appends a subset of elements from one buffer to another.
     * @details The elements to append are specified by indices in `idxs_buf`.
     * The source buffer `buf` is treated as an array of objects, each with `nvar` variables.
     * The elements are appended to `buf_other`.
     *
     * @tparam T The type of data in the buffers.
     * @param buf The source buffer.
     * @param idxs_buf A buffer of indices specifying which objects to copy from `buf`.
     * @param nvar The number of variables per object.
     * @param buf_other The destination buffer to which the subset will be appended.
     */
    template<class T>
    inline void append_subset_to(
        const sham::DeviceBuffer<T> &buf,
        const sham::DeviceBuffer<u32> &idxs_buf,
        u32 nvar,
        sham::DeviceBuffer<T> &buf_other) {

        const u64 idx_count = idxs_buf.get_size();

        if (idx_count == 0) {
            return;
        }

        u64 idx_to_insert = shambase::narrow_or_throw<u32>(idx_count);

        u32 start_insert_idx = shambase::narrow_or_throw<u32>(buf_other.get_size());

        buf_other.expand(shambase::narrow_or_throw<u32>(idx_count * nvar));

        sham::kernel_call(
            idxs_buf.get_queue(),
            sham::MultiRef{idxs_buf, buf},
            sham::MultiRef{buf_other},
            idx_to_insert,
            [nvar_loc = nvar, start_insert_idx_loc = start_insert_idx](
                u32 gid,
                const u32 *__restrict acc_idxs,
                const T *__restrict acc_curr,
                T *__restrict acc_other) {
                u32 idx_extr = acc_idxs[gid] * nvar_loc;
                u32 idx_push = start_insert_idx_loc + gid * nvar_loc;

                for (u32 a = 0; a < nvar_loc; a++) {
                    acc_other[idx_push + a] = acc_curr[idx_extr + a];
                }
            });
    }

} // namespace shamalgs::primitives
