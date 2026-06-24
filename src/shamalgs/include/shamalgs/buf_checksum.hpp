// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file buf_checksum.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/checksum.hpp"
#include "shamalgs/primitives/flatten.hpp"
#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs {

    template<class T>
    inline u64 buf_checksum(const sham::DeviceBuffer<T> &buf) {
        auto flattened_buf = primitives::flatten_buffer(buf);

        using Tscal             = typename shambase::VectorProperties<T>::component_type;
        std::vector<Tscal> data = flattened_buf.copy_to_stdvec();
        return shambase::fnv1a_hash(
            reinterpret_cast<const char *>(data.data()), data.size() * sizeof(Tscal));
    }

} // namespace shamalgs
