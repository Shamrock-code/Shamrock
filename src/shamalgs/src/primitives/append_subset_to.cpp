// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file append_subset_to.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

#define XMAC_LIST_ENABLED_FIELD                                                                    \
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

namespace shamalgs::primitives {

    template<class T>
    void append_subset_to(
        const sham::DeviceBuffer<T> &buf,
        const sham::DeviceBuffer<u32> &idxs_buf,
        u32 nvar,
        sham::DeviceBuffer<T> &buf_other,
        u32 start_enque) {

        const u64 idx_count = idxs_buf.get_size();

        if (idx_count == 0) {
            return;
        }

        // TODO: flatten the kernel to squeeze more perf
        sham::kernel_call(
            idxs_buf.get_queue(),
            sham::MultiRef{idxs_buf, buf},
            sham::MultiRef{buf_other},
            shambase::narrow_or_throw<u32>(idx_count),
            [nvar_loc = nvar, start_enque_loc = start_enque](
                u32 gid,
                const u32 *__restrict acc_idxs,
                const T *__restrict acc_curr,
                T *__restrict acc_other) {
                u32 idx_extr = acc_idxs[gid] * nvar_loc;
                u32 idx_push = start_enque_loc + gid * nvar_loc;

                for (u32 a = 0; a < nvar_loc; a++) {
                    acc_other[idx_push + a] = acc_curr[idx_extr + a];
                }
            });
    }

#ifndef DOXYGEN
    #define X(a)                                                                                   \
        template void append_subset_to<a>(                                                         \
            const sham::DeviceBuffer<a> &buf,                                                      \
            const sham::DeviceBuffer<u32> &idxs_buf,                                               \
            u32 nvar,                                                                              \
            sham::DeviceBuffer<a> &buf_other,                                                      \
            u32 start_enque);

    XMAC_LIST_ENABLED_FIELD
    #undef X
#endif
} // namespace shamalgs::primitives
