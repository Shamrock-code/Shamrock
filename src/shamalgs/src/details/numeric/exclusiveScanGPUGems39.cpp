// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file exclusiveScanGPUGems39.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"

/*

GPU GEMS chapter 39
(https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

ALG 1 :
__global__ void scan_1(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int pout = 0, pin = 1; // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        if (thid >= offset)
            temp[pout * n + thid] += temp[pin * n + thid - offset];
        else
            temp[pout * n + thid] = temp[pin * n + thid];
        __syncthreads();
    }
    g_odata[thid] = temp[pout * n + thid]; // write output
}

*/

namespace shamalgs::numeric::details {
    template<class T>
    class KernelExclsum_1;

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_1(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        sycl::buffer<T> out1(len);
        sycl::buffer<T> out2(len);

        auto get_in_buf_ref = [&](u32 step) -> sycl::buffer<T> & {
            if (step % 2 == 0) {
                return out1;
            } else {
                return out2;
            }
        };

        auto get_out_buf_ref = [&](u32 step) -> sycl::buffer<T> & {
            if (step % 2 == 1) {
                return out1;
            } else {
                return out2;
            }
        };

        u32 step = 0;

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{get_in_buf_ref(step), cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0) ? acc_in[thid - 1] : 0;
            });
        });

        for (int offset = 1; offset < len; offset *= 2) {

            q.submit([&, offset](sycl::handler &cgh) {
                sycl::accessor acc_in{get_in_buf_ref(step), cgh, sycl::read_only};
                sycl::accessor acc_out{get_out_buf_ref(step), cgh, sycl::write_only};

                cgh.parallel_for<KernelExclsum_1<T>>(sycl::range<1>{len}, [=](sycl::item<1> id) {
                    u32 thid = id.get_linear_id();

                    const auto in_val = acc_in[thid];

                    acc_out[thid] = (thid >= offset) ? in_val + acc_in[thid - offset] : in_val;
                });
            });

            step++;
        }

        return std::move(get_in_buf_ref(step));
    }

    template<class T>
    class KernelExclsum_2;

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_2(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 rounded_len = sham::roundup_pow2_clz(len);

        sycl::buffer<T> out1(rounded_len);
        sycl::buffer<T> out2(rounded_len);

        auto get_in_buf_ref = [&](u32 step) -> sycl::buffer<T> & {
            if (step % 2 == 0) {
                return out1;
            } else {
                return out2;
            }
        };

        auto get_out_buf_ref = [&](u32 step) -> sycl::buffer<T> & {
            if (step % 2 == 1) {
                return out1;
            } else {
                return out2;
            }
        };

        u32 step = 0;

        q.submit([&](sycl::handler &cgh) {
            u32 corect_len = len;
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{get_in_buf_ref(step), cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{rounded_len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0 && thid < corect_len) ? acc_in[thid - 1] : 0;
            });
        });

        for (int offset = 1; offset < rounded_len; offset *= 2) {

            q.submit([&, offset](sycl::handler &cgh) {
                sycl::accessor acc_in{get_in_buf_ref(step), cgh, sycl::read_only};
                sycl::accessor acc_out{get_out_buf_ref(step), cgh, sycl::write_only};

                cgh.parallel_for<KernelExclsum_2<T>>(
                    sycl::range<1>{rounded_len}, [=](sycl::item<1> id) {
                        u32 thid = id.get_linear_id();

                        const auto in_val = acc_in[thid];

                        acc_out[thid] = (thid >= offset) ? in_val + acc_in[thid - offset] : in_val;
                    });
            });

            step++;
        }

        return std::move(get_in_buf_ref(step));
    }

    template<class T>
    class KernelExclsum_3;

    template<class T>
    sycl::buffer<T> exclusive_sum_gpugems39_3(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {

        u32 rounded_len = sham::roundup_pow2_clz(len);

        sycl::buffer<T> out1(rounded_len);
        sycl::buffer<T> out2(rounded_len);

        auto get_in_buf_ref = [&](u32 step) -> sycl::buffer<T> & {
            if (step % 2 == 0) {
                return out1;
            } else {
                return out2;
            }
        };

        auto get_out_buf_ref = [&](u32 step) -> sycl::buffer<T> & {
            if (step % 2 == 1) {
                return out1;
            } else {
                return out2;
            }
        };

        u32 step = 0;

        q.submit([&](sycl::handler &cgh) {
            u32 corect_len = len;
            sycl::accessor acc_in{buf1, cgh, sycl::read_only};
            sycl::accessor acc_out{get_in_buf_ref(step), cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{rounded_len}, [=](sycl::item<1> id) {
                u32 thid    = id.get_linear_id();
                acc_out[id] = (thid > 0 && thid < corect_len) ? acc_in[thid - 1] : 0;
            });
        });

        for (int offset = 1; offset < rounded_len; offset *= 2) {

            q.submit([&, offset](sycl::handler &cgh) {
                sycl::accessor acc_in{get_in_buf_ref(step), cgh, sycl::read_only};
                sycl::accessor acc_out{get_out_buf_ref(step), cgh, sycl::write_only};

                cgh.parallel_for<KernelExclsum_3<T>>(
                    sycl::range<1>{rounded_len}, [=](sycl::item<1> id) {
                        u32 thid = id.get_linear_id();

                        const auto in_val = acc_in[thid];

                        acc_out[thid] = (thid >= offset) ? in_val + acc_in[thid - offset] : in_val;
                    });
            });

            step++;
        }

        return std::move(get_in_buf_ref(step));
    }

    template sycl::buffer<u32>
    exclusive_sum_gpugems39_1(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template sycl::buffer<u32>
    exclusive_sum_gpugems39_2(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template sycl::buffer<u32>
    exclusive_sum_gpugems39_3(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

} // namespace shamalgs::numeric::details
