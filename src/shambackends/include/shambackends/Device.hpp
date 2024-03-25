// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RadixTree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"

namespace sham {

    enum class Vendor{
        UNKNOWN ,NVIDIA, AMD, INTEL, APPLE
    };

    enum class Backend{
        UNKNOWN ,CUDA, ROCM, OPENMP
    };

    struct DeviceProperties{
        Vendor vendor;
        Backend backend;
    };

    struct DeviceMPIProperties{
        bool is_mpi_direct_capable;
    };

    class Device{
        public:

        usize device_id;

        sycl::device dev;
        
        DeviceProperties prop;

        DeviceMPIProperties mpi_prop;

    };

    std::vector<std::unique_ptr<Device>> get_device_list();

}