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

#include "shambase/exception.hpp"
#include "shambackends/sycl.hpp"

namespace sham {

    enum class Vendor { UNKNOWN, NVIDIA, AMD, INTEL, APPLE };

    inline std::string vendor_name(Vendor v){
        switch (v) {
            case Vendor::UNKNOWN : return "Unknown";
            case Vendor::NVIDIA : return "Nvidia";
            case Vendor::AMD : return "AMD";
            case Vendor::INTEL : return "Intel";
            case Vendor::APPLE : return "Apple";
            default: shambase::throw_unimplemented();
        }
    }

    enum class Backend { UNKNOWN, CUDA, ROCM, OPENMP };

    inline std::string backend_name(Backend b){
        switch (b) {
            case Backend::UNKNOWN : return "Unknown";
            case Backend::CUDA : return "CUDA";
            case Backend::ROCM : return "ROCM";
            case Backend::OPENMP : return "OpenMP";
            default: shambase::throw_unimplemented();
        }
    }

    struct DeviceProperties {
        Vendor vendor;
        Backend backend;
    };

    struct DeviceMPIProperties {
        bool is_mpi_direct_capable;
    };

    class Device {
        public:
        usize device_id;

        sycl::device dev;

        DeviceProperties prop;

        DeviceMPIProperties mpi_prop;

        void update_mpi_prop();
    };

    std::vector<std::unique_ptr<Device>> get_device_list();

    Device sycl_dev_to_sham_dev(usize i, const sycl::device &dev);

} // namespace sham