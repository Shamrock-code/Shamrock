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

#include "shambackends/DeviceBuffer.hpp"

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
    void append_subset_to(
        const sham::DeviceBuffer<T> &buf,
        const sham::DeviceBuffer<u32> &idxs_buf,
        u32 nvar,
        sham::DeviceBuffer<T> &buf_other);

} // namespace shamalgs::primitives
