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
 * @file dot_sum.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::primitives {

    template<class T>
    shambase::VecComponent<T> dot_sum(sham::DeviceBuffer<T> &buf1, u32 start_id, u32 end_id);

    template<class T>
    inline shambase::VecComponent<T> dot_sum(sham::DeviceBuffer<T> &buf1, u32 start_id) {
        return dot_sum(buf1, start_id, buf1.get_size());
    }

    template<class T>
    inline shambase::VecComponent<T> dot_sum(sham::DeviceBuffer<T> &buf1) {
        return dot_sum(buf1, 0, buf1.get_size());
    }

} // namespace shamalgs::primitives
