// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <sycl/sycl.hpp>

int main(void) {
    const int N = 10000;
    sycl::buffer<int> buf_int(10000);

    sycl::buffer<int> recov{1};

    sycl::queue{}.submit([&](sycl::handler &cgh) {
        sycl::accessor global_mem{buf_int, cgh, sycl::read_only};

        auto reduc = sycl::reduction(recov, cgh, sycl::plus<>{});

        cgh.parallel_for(sycl::range<1>{N}, reduc, [=](sycl::id<1> idx, auto &sum) {
            sum.combine(global_mem[idx]);
        });
    });

    int rec;
    {
        sycl::host_accessor acc{recov, sycl::read_only};
        rec = acc[0];
    }
}
