// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file memoryHandle.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
 
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/details/BufferEventHandler.hpp"

namespace sham::details {

    template<USMKindTarget target>
    USMPtrHolder<target> create_usm_ptr(u32 size, std::shared_ptr<DeviceScheduler> dev_sched);

    template<USMKindTarget target>
    void release_usm_ptr(USMPtrHolder<target> &&usm_ptr_hold, details::BufferEventHandler &&events);

} // namespace sham::details
