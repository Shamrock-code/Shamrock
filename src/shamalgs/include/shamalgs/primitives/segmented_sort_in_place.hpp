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
 * @file segmented_sort_in_place.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamalgs/impl_utils.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives {

    template<class T>
    void segmented_sort_in_place(
        sham::DeviceBuffer<T> &buf, const sham::DeviceBuffer<u32> &offsets);

    /// namespace to control implementation behavior
    namespace impl {

        /// Get list of available segmented sort in place implementations
        std::vector<shamalgs::impl_param> get_default_impl_list_segmented_sort_in_place();

        /// Get the current implementation for segmented sort in place
        shamalgs::impl_param get_current_impl_segmented_sort_in_place();

        /// Set the implementation for segmented sort in place
        void set_impl_segmented_sort_in_place(
            const std::string &impl, const std::string &param = "");

    } // namespace impl

} // namespace shamalgs::primitives
