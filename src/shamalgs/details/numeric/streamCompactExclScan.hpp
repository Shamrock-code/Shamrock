// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file streamCompactExclScan.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "aliases.hpp"
#include "shambase/sycl.hpp"

namespace shamalgs::numeric::details {


    /**
     * @brief Stream compaction algorithm using exclusive summation
     * 
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return sycl::buffer<u32> table of the index to extract
     */
    std::tuple<std::optional<sycl::buffer<u32>>, u32> stream_compact_excl_scan(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

    
} // namespace shamalgs::numeric::details