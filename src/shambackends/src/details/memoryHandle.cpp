// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file memoryHandle.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/USMPtrHolder.hpp"
#include <shambackends/details/memoryHandle.hpp>

namespace sham::details {

    template<>
    USMPtrHolder<device>
    create_usm_ptr<device>(u32 size, std::shared_ptr<DeviceScheduler> dev_sched) {

        std::cout << "create usm pointer size : " << size << " | mode = device" << std::endl;
        auto ret = USMPtrHolder<device>(size, dev_sched);
        std::cout << "pointer created : ptr = " << ret.get_raw_ptr() << std::endl;
        return ret;
    }

    template<>
    USMPtrHolder<shared>
    create_usm_ptr<shared>(u32 size, std::shared_ptr<DeviceScheduler> dev_sched) {

        std::cout << "create usm pointer size : " << size << " | mode = shared" << std::endl;
        auto ret = USMPtrHolder<shared>(size, dev_sched);
        std::cout << "pointer created : ptr = " << ret.get_raw_ptr() << std::endl;
        return ret;
    }

    template<>
    USMPtrHolder<host> create_usm_ptr<host>(u32 size, std::shared_ptr<DeviceScheduler> dev_sched) {

        std::cout << "create usm pointer size : " << size << " | mode = host" << std::endl;
        auto ret = USMPtrHolder<host>(size, dev_sched);
        std::cout << "pointer created : ptr = " << ret.get_raw_ptr() << std::endl;
        return ret;
    }

    template<>
    void release_usm_ptr<device>(USMPtrHolder<device> &&usm_ptr_hold, BufferEventHandler &&events) {

        std::cout << "release usm pointer size : " << usm_ptr_hold.get_size()
                  << " | ptr = " << usm_ptr_hold.get_raw_ptr() << " | mode = device"
                  << std::endl;
        std::cout << "waiting event completion ..." << std::endl;
        events.wait_all();
        std::cout << "done, freeing memory" << std::endl;
        usm_ptr_hold.free_ptr();
    }

    template<>
    void release_usm_ptr<shared>(USMPtrHolder<shared> &&usm_ptr_hold, BufferEventHandler &&events) {

        std::cout << "release usm pointer size : " << usm_ptr_hold.get_size()
                  << " | ptr = " << usm_ptr_hold.get_raw_ptr() << " | mode = shared"
                  << std::endl;
        std::cout << "waiting event completion ..." << std::endl;
        events.wait_all();
        std::cout << "done, freeing memory" << std::endl;
        usm_ptr_hold.free_ptr();
    }

    template<>
    void release_usm_ptr<host>(USMPtrHolder<host> &&usm_ptr_hold, BufferEventHandler &&events) {

        std::cout << "release usm pointer size : " << usm_ptr_hold.get_size()
                  << " | ptr = " << usm_ptr_hold.get_raw_ptr() << " | mode = host"
                  << std::endl;
        std::cout << "waiting event completion ..." << std::endl;
        events.wait_all();
        std::cout << "done, freeing memory" << std::endl;
        usm_ptr_hold.free_ptr();
    }

} // namespace sham::details