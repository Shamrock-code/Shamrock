// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once
/**
 * @file buffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#ifdef SHAMROCK_ENABLE_BACKEND_SYCL

#include "sycl.hpp"

namespace sham::details {

    template<class T>
    class buffer {

        std::unique_ptr<sycl::buffer<T>> storage;

        inline std::unique_ptr<sycl::buffer<T>> &get_native_handle() { return storage; }
    };
    
} // namespace sham::details

#endif