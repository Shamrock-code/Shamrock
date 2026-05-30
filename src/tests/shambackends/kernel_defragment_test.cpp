// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/SourceLocation.hpp"
#include "shambase/narrowing.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/benchmarks/saxpy.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <utility>
#include <vector>

struct bufs_test {
    std::vector<sham::DeviceBuffer<f32>> bufs_x;
    std::vector<sham::DeviceBuffer<f32>> bufs_y;

    static constexpr f32 init_x     = 2.0f;
    static constexpr f32 init_y     = 5.0f;
    static constexpr f32 a          = 6.0f;
    static constexpr f32 expected_y = a * init_x + init_y;

    void make_test_bufs(u32 n_bufs, u32 size) {

        bufs_x.clear();
        bufs_y.clear();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        for (size_t i = 0; i < n_bufs; i++) {
            bufs_x.push_back(sham::DeviceBuffer<f32>{size, dev_sched});
            bufs_y.push_back(sham::DeviceBuffer<f32>{size, dev_sched});
        }
    }

    void fill_bufs() {
        for (size_t i = 0; i < bufs_x.size(); i++) {
            bufs_x[i].fill(init_x);
            bufs_y[i].fill(init_y);
        }
        synchronize_bufs();
    }

    void synchronize_bufs() {
        for (size_t i = 0; i < bufs_x.size(); i++) {
            bufs_x[i].synchronize();
            bufs_y[i].synchronize();
        }
    }

    bool check_correctness() {
        for (size_t i = 0; i < bufs_x.size(); i++) {
            bufs_y[i].synchronize();
        }

        for (size_t i = 0; i < bufs_x.size(); i++) {
            auto actual_y = bufs_y[i].copy_to_stdvec();
            for (size_t j = 0; j < actual_y.size(); j++) {
                if (actual_y[j] != expected_y) {
#ifdef false
                    logger::raw_ln(
                        shambase::format(
                            " - buf_y[{}][{}] = {} != expected_y = {}",
                            i,
                            j,
                            bufs_y[i].get_val_at_idx(j),
                            expected_y));
#endif
                    return false;
                }
            }
        }
        return true;
    }
};

void saxpy_many_kernels_base_impl(
    sham::DeviceQueue &q,
    sham::DeviceScheduler_ptr dev_sched,
    f32 a,
    std::vector<sham::DeviceBuffer<f32>> &x,
    std::vector<sham::DeviceBuffer<f32>> &y) {

    for (size_t i = 0; i < x.size(); i++) {
        sham::kernel_call(
            q,
            sham::MultiRef{x[i]},
            sham::MultiRef{y[i]},
            shambase::narrow_or_throw<u32>(x[i].get_size()),
            [a](u32 i, const f32 *__restrict x, f32 *__restrict y) {
                y[i] = a * x[i] + y[i];
            });
    }
}

void saxpy_many_kernels_base_fuse_basic(
    sham::DeviceQueue &q,
    sham::DeviceScheduler_ptr dev_sched,
    f32 a,
    std::vector<sham::DeviceBuffer<f32>> &x,
    std::vector<sham::DeviceBuffer<f32>> &y) {

    constexpr u32 group_size = 256;

    sham::EventList depends_list;

    struct params {
        const f32 *x;
        f32 *y;
        size_t size;
    };

    u32 ngroups = 0;
    std::vector<params> params_vec(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        u32 gcount = shambase::group_count(x[i].get_size(), group_size);
        ngroups += gcount;
        params_vec[i] = params{
            x[i].get_read_access(depends_list),
            y[i].get_write_access(depends_list),
            x[i].get_size()};
    }

    sham::DeviceBuffer<params> params_buf{params_vec.size(), dev_sched};
    params_buf.copy_from_stdvec(params_vec);

    const params *params_ptr = params_buf.get_read_access(depends_list);

    u32 corrected_len = ngroups * group_size;

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        using atomic_ref_u32 = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>;

        sycl::local_accessor<params> local_params{1, h};
        sycl::local_accessor<u32> local_group_index{1, h};
        u32 param_count = params_vec.size();

        h.parallel_for(sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
            bool is_main_thread = id.get_local_id(0) == 0;

            if (is_main_thread) {
                u32 current_group_id = id.get_group_linear_id();
                params current_params;

                // move up the params until the index is found
                for (size_t i = 0; i < param_count; i++) {
                    current_params = params_ptr[i];

                    u32 group_count_param = shambase::group_count(current_params.size, group_size);

                    if (current_group_id < group_count_param) {
                        local_params[0]      = current_params;
                        local_group_index[0] = current_group_id;
                        break;
                    }

                    current_group_id -= group_count_param;
                }
            }

            id.barrier(sycl::access::fence_space::local_space);

            params current_params   = local_params[0];
            u32 current_group_index = local_group_index[0];

            u32 tile_id = current_group_index * group_size + id.get_local_id(0);

            current_params.y[tile_id] = a * current_params.x[tile_id] + current_params.y[tile_id];
        });
    });

    params_buf.complete_event_state(e);

    for (size_t i = 0; i < x.size(); i++) {
        x[i].complete_event_state(e);
        y[i].complete_event_state(e);
    }
}

void saxpy_many_kernels_base_fuse_sycl_buffer(
    sham::DeviceQueue &q,
    sham::DeviceScheduler_ptr dev_sched,
    f32 a,
    std::vector<sham::DeviceBuffer<f32>> &x,
    std::vector<sham::DeviceBuffer<f32>> &y) {

    constexpr u32 group_size = 256;

    sham::EventList depends_list;

    struct params {
        const f32 *x;
        f32 *y;
        size_t size;
    };

    u32 ngroups = 0;
    std::vector<params> params_vec(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        u32 gcount = shambase::group_count(x[i].get_size(), group_size);
        ngroups += gcount;
        params_vec[i] = params{
            x[i].get_read_access(depends_list),
            y[i].get_write_access(depends_list),
            x[i].get_size()};
    }

    sycl::buffer<params> params_buf{params_vec};

    u32 corrected_len = ngroups * group_size;

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        using atomic_ref_u32 = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>;

        sycl::local_accessor<params> local_params{1, h};
        sycl::local_accessor<u32> local_group_index{1, h};
        u32 param_count = params_vec.size();

        sycl::accessor params_acc{params_buf, h, sycl::read_only};

        h.parallel_for(sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
            bool is_main_thread = id.get_local_id(0) == 0;

            if (is_main_thread) {
                u32 current_group_id = id.get_group_linear_id();
                params current_params;

                // move up the params until the index is found
                for (size_t i = 0; i < param_count; i++) {
                    current_params = params_acc[i];

                    u32 group_count_param = shambase::group_count(current_params.size, group_size);

                    if (current_group_id < group_count_param) {
                        local_params[0]      = current_params;
                        local_group_index[0] = current_group_id;
                        break;
                    }

                    current_group_id -= group_count_param;
                }
            }

            id.barrier(sycl::access::fence_space::local_space);

            params current_params   = local_params[0];
            u32 current_group_index = local_group_index[0];

            u32 tile_id = current_group_index * group_size + id.get_local_id(0);

            current_params.y[tile_id] = a * current_params.x[tile_id] + current_params.y[tile_id];
        });
    });

    for (size_t i = 0; i < x.size(); i++) {
        x[i].complete_event_state(e);
        y[i].complete_event_state(e);
    }
}

void saxpy_many_kernels_base_fuse_usm(
    sham::DeviceQueue &q,
    sham::DeviceScheduler_ptr dev_sched,
    f32 a,
    std::vector<sham::DeviceBuffer<f32>> &x,
    std::vector<sham::DeviceBuffer<f32>> &y) {

    constexpr u32 group_size = 256;

    sham::EventList depends_list;

    struct params {
        const f32 *x;
        f32 *y;
        size_t size;
    };

    u32 ngroups = 0;
    std::vector<params> params_vec(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        u32 gcount = shambase::group_count(x[i].get_size(), group_size);
        ngroups += gcount;
        params_vec[i] = params{
            x[i].get_read_access(depends_list),
            y[i].get_write_access(depends_list),
            x[i].get_size()};
    }

    params *params_acc = sycl::malloc_device<params>(params_vec.size(), q.q);
    auto e_cpy = q.q.memcpy(params_acc, params_vec.data(), params_vec.size() * sizeof(params));
    depends_list.add_event(e_cpy);

    u32 corrected_len = ngroups * group_size;

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        using atomic_ref_u32 = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>;

        sycl::local_accessor<params> local_params{1, h};
        sycl::local_accessor<u32> local_group_index{1, h};
        u32 param_count = params_vec.size();

        h.parallel_for(sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
            bool is_main_thread = id.get_local_id(0) == 0;

            if (is_main_thread) {
                u32 current_group_id = id.get_group_linear_id();
                params current_params;

                // move up the params until the index is found
                for (size_t i = 0; i < param_count; i++) {
                    current_params = params_acc[i];

                    u32 group_count_param = shambase::group_count(current_params.size, group_size);

                    if (current_group_id < group_count_param) {
                        local_params[0]      = current_params;
                        local_group_index[0] = current_group_id;
                        break;
                    }

                    current_group_id -= group_count_param;
                }
            }

            id.barrier(sycl::access::fence_space::local_space);

            params current_params   = local_params[0];
            u32 current_group_index = local_group_index[0];

            u32 tile_id = current_group_index * group_size + id.get_local_id(0);

            current_params.y[tile_id] = a * current_params.x[tile_id] + current_params.y[tile_id];
        });
    });

    e.wait();
    sycl::free(params_acc, q.q);

    for (size_t i = 0; i < x.size(); i++) {
        x[i].complete_event_state(sycl::event{});
        y[i].complete_event_state(sycl::event{});
    }
}

template<class TestFunc>
void testing_func_kernel_defragmentation_base(
    TestFunc test_func,
    sham::DeviceQueue &q,
    sham::DeviceScheduler_ptr dev_sched,
    u32 n_bufs,
    u32 size) {
    __shamrock_stack_entry();
    bufs_test bufs;
    bufs.make_test_bufs(n_bufs, size);
    bufs.fill_bufs();

    f64 min_time = std::numeric_limits<f64>::max();

    for (u32 i = 0; i < 10; i++) {
        bufs.fill_bufs();

        shambase::Timer timer;
        timer.start();

        test_func(q, dev_sched, bufs.a, bufs.bufs_x, bufs.bufs_y);

        bufs.synchronize_bufs();

        timer.end();
        min_time = std::min(min_time, timer.elasped_sec());
    }

    bool correct = bufs.check_correctness();

    REQUIRE(correct);

    std::string correct_str = correct ? "correct" : "incorrect";

    logger::raw_ln(
        shambase::format(
            " - n_bufs: {:4d} - size: {:6d} - time: {:.4e} seconds - correctness: {}",
            n_bufs,
            size,
            min_time,
            correct_str));
}

TestStart(Unittest, "kernel_defragmentation", testing_func_kernel_defragmentation_base, 1) {
    sham::DeviceScheduler_ptr dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q                = dev_sched->get_queue();

    sham::DeviceQueue q_inorder = sham::DeviceQueue("inorder_queue", q.ctx, true);

    logger::raw_ln("Testing base case (out of order):");
    u32 total_size    = 1024 * 256;
    u32 min_fragments = 1;
    u32 max_fragments = 2048;
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_impl, q, dev_sched, fragments, total_size / fragments);
    }

    logger::raw_ln("Testing base case (in order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_impl, q_inorder, dev_sched, fragments, total_size / fragments);
    }

    logger::raw_ln("Testing base case (fuse lock out of order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_fuse_basic, q, dev_sched, fragments, total_size / fragments);
    }

    logger::raw_ln("Testing base case (fuse lock in order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_fuse_basic,
            q_inorder,
            dev_sched,
            fragments,
            total_size / fragments);
    }

    logger::raw_ln("Testing base case (fuse sycl buffer out of order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_fuse_sycl_buffer,
            q,
            dev_sched,
            fragments,
            total_size / fragments);
    }

    logger::raw_ln("Testing base case (fuse sycl buffer in order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_fuse_sycl_buffer,
            q_inorder,
            dev_sched,
            fragments,
            total_size / fragments);
    }

    logger::raw_ln("Testing base case (fuse usm out of order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_fuse_usm, q, dev_sched, fragments, total_size / fragments);
    }

    logger::raw_ln("Testing base case (fuse usm in order):");
    for (u32 fragments = min_fragments; fragments <= max_fragments; fragments *= 2) {
        testing_func_kernel_defragmentation_base(
            saxpy_many_kernels_base_fuse_usm,
            q_inorder,
            dev_sched,
            fragments,
            total_size / fragments);
    }
}
