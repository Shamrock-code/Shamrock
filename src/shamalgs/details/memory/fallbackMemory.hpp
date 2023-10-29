// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file fallbackMemory.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "aliases.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::memory::details {

    template<class T>
    struct Fallback{

        static T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

        
        static sycl::buffer<T> vec_to_buf(const std::vector<T> &vec);
        static std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len);

    };
    

    


}