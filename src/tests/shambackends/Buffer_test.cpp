// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


/**
 * @file math.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/Buffer.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "sham::Buffer", sham_buffer, 1){

    sham::Buffer<f64> buf;

    _Assert(buf.get_capacity() == 0)
    _Assert(buf.is_allocated() == false)
    buf.alloc(1000);

    _Assert(buf.get_capacity() >= 1000)
    _Assert(buf.is_allocated() == true)

    buf.free();
    _Assert(buf.get_capacity() == 0)
    _Assert(buf.is_allocated() == false)

}