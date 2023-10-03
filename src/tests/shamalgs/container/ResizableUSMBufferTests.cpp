// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/container/ResizableUSMBuffer.hpp"
#include <shambackends/backends/sycl/sycl.hpp>
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>

TestStart(Unittest, "shamalgs/container/ResizableUSMBuffer", test_resizableUSMBuffer, 1) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    {
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Host);
        usm_rbuf.change_capacity(100);
    }

    {
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Shared);
        usm_rbuf.change_capacity(100);
    }

    {
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Device);
        usm_rbuf.change_capacity(100);
    }

    { // test move constructor to check if no double free are performed

        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Host);
        usm_rbuf.change_capacity(100);
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf2{std::move(usm_rbuf)};

    }

}
