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

    sycl::queue{}.submit([&](sycl::handler &cgh) {
        sycl::accessor global_mem{buf_int, cgh, sycl::read_only};

        cgh.parallel_for(sycl::nd_range<1>{N, 10}, [=](sycl::nd_item<1> item) {
            auto ret = sycl::reduce_over_group(
                item.get_group(), global_mem[item.get_global_linear_id()], sycl::plus<>{});
        });
    });
}
