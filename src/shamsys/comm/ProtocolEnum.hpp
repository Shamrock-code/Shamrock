// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ProtocolEnum.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

namespace shamsys::comm {

    enum Protocol{
        /**
         * @brief copy data to the host and then perform the call
         */
        CopyToHost, 
        
        /**
         * @brief copy data straight from the GPU
         */
        DirectGPU, 
        
        /**
         * @brief  copy data straight from the GPU & flatten sycl vector to plain arrays
         */
        DirectGPUFlatten,
    };
}