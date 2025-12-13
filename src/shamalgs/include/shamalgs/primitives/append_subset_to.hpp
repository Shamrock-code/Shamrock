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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives {

    template<class T>
    inline void append_subset_to(
        const sham::DeviceBuffer<T> &buf,
        const sham::DeviceBuffer<u32> &idxs_buf,
        u32 nvar,
        sham::DeviceBuffer<T> &buf_other) {

        auto &q = idxs_buf.get_queue();

        u64 idx_to_insert    = idxs_buf.get_size();
        buf_other.expand(idx_to_insert * nvar);

        u64 start_insert_idx = buf_other.get_size();

        sham::kernel_call(
            q,
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
