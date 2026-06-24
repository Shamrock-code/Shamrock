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
 * @file PatchField.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"
#include <memory>
#include <optional>
namespace shamrock::patch {

    template<class T>
    class PatchField {
        public:
        shambase::DistributedData<T> field_all;

        PatchField(shambase::DistributedData<T> &&field_all) : field_all(field_all) {}

        T &get(u64 id) { return field_all.get(id); }
    };

    template<class T>
    class PatchtreeField {
        public:
        std::optional<sham::DeviceBuffer<T>> internal_buf;

        inline void reset() { internal_buf.reset(); }

        inline void allocate(u32 size, sham::DeviceScheduler_ptr sched) {
            internal_buf.emplace(size, sched);
        }
    };
} // namespace shamrock::patch
