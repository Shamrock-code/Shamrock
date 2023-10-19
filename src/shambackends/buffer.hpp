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
#include "shambackends/backends/sycl/buffer.hpp"
#include <utility>
#ifdef SHAMBACKENDS_USE_SYCL
    #include <shambackends/backends/sycl/sycl.hpp>
#endif
#ifdef SHAMBACKENDS_USE_KOKKOS
    #include <shambackends/backends/kokkos/kokkos.hpp>
#endif

namespace sham {

    template<class T>
    using buffer = details::buffer<T>;

}