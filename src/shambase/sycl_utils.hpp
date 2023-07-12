// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "sycl_utils/sycl_utilities.hpp"
#include "sycl_utils/vec_equals.hpp"
#include "sycl_utils/vectorProperties.hpp"
#include <stdexcept>

namespace shambase {

    /**
     * @brief check that the size of a sycl buffer is below or equal to the value of max range
     * throw if it is not the case
     *
     * @tparam T
     * @param buf
     * @param max_range
     * @param loc
     */
    template<class T>
    void check_buffer_size(sycl::buffer<T> &buf,
                           u64 max_range,
                           struct SourceLocation loc = SourceLocation()) {
        if (buf.size() < max_range) {
            throw throw_with_loc<std::invalid_argument>("buffer is too small", loc);
        }
    }

    /**
     * @brief Get the Device Type Name
     *
     * @param Device
     * @return std::string
     */
    inline std::string getDevice_type(const sycl::device &Device) {
        auto DeviceType = Device.get_info<sycl::info::device::device_type>();
        switch (DeviceType) {
        case sycl::info::device_type::cpu: return "CPU";
        case sycl::info::device_type::gpu: return "GPU";
        case sycl::info::device_type::host: return "HOST";
        case sycl::info::device_type::accelerator: return "ACCELERATOR";
        default: return "UNKNOWN";
        }
    }

    /**
     * @brief Generate a sycl nd range out of a group size and lenght
     * 
     * @param lenght max index value
     * @param group_size group size
     * @return sycl::nd_range<1> the sycl nd range
     */
    inline sycl::nd_range<1> make_range(u32 lenght, const u32 group_size = 32){
        u32 group_cnt = shambase::group_count(lenght, group_size);
        u32 len =group_cnt*group_size;
        return sycl::nd_range<1>{len, group_size};
    }

} // namespace shambase