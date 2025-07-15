// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ResizableUSMBuffer.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/container/ResizableUSMBuffer.hpp"
#include "shamalgs/details/reduction/reduction.hpp"
#include "shamcomm/logs.hpp"

template<class T>
void shamalgs::ResizableUSMBuffer<T>::alloc() {
    if (usm_ptr != nullptr) {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "the usm pointer is already allocated");
    }

    if (type == Host) {
        usm_ptr = sycl::malloc_host<T>(capacity, q);
        shamlog_debug_alloc_ln(
            "ResizableBufferUSM", events_hndl.get_hash_log(), "alloc HOST N =", capacity);
    } else if (type == Device) {
        usm_ptr = sycl::malloc_device<T>(capacity, q);
        shamlog_debug_alloc_ln(
            "ResizableBufferUSM", events_hndl.get_hash_log(), "alloc DEVICE N =", capacity);
    } else if (type == Shared) {
        usm_ptr = sycl::malloc_shared<T>(capacity, q);
        shamlog_debug_alloc_ln(
            "ResizableBufferUSM", events_hndl.get_hash_log(), "alloc SHARED N =", capacity);
    }
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::free() {
    StackEntry stack_loc{};
    events_hndl.synchronize();

    shamlog_debug_alloc_ln("ResizableBufferUSM", events_hndl.get_hash_log(), "free");
    sycl::free(usm_ptr, q);
    usm_ptr = nullptr;
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::change_capacity(u32 new_capa) {
    StackEntry stack_loc{false};

    shamlog_debug_alloc_ln(
        "ResizableBufferUSM",
        events_hndl.get_hash_log(),
        "change capacity from : ",
        capacity,
        "to :",
        new_capa);

    if (capacity == 0) {

        if (new_capa > 0) {
            capacity = new_capa;
            alloc();
        }

    } else {

        if (new_capa > 0) {

            if (new_capa != capacity) {
                capacity = new_capa;

                T *old_usm = release_usm_ptr();

                alloc();

                if (val_count > 0) {
                    q.memcpy(usm_ptr, old_usm, sizeof(T) * val_count).wait();
                }

                shamlog_debug_alloc_ln(
                    "ResizableBufferUSM", events_hndl.get_hash_log(), "delete old buf");
                sycl::free(old_usm, q);
            }

        } else {
            capacity = 0;
            free();
        }
    }
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::reserve(u32 add_size) {
    StackEntry stack_loc{false};

    u32 wanted_sz = val_count + add_size;

    if (wanted_sz > capacity) {
        change_capacity(wanted_sz * safe_fact);
    }
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::resize(u32 new_size) {
    StackEntry stack_loc{false};

    shamlog_debug_alloc_ln(
        "ResizableBufferUSM",
        events_hndl.get_hash_log(),
        "resize from : ",
        val_count,
        "to :",
        new_size);

    if (new_size > capacity) {
        change_capacity(new_size * safe_fact);
    }

    val_count = new_size;
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::change_buf_type(BufferType new_type) {

    type = new_type;

    T *old_usm = release_usm_ptr();

    alloc();

    if (val_count > 0) {
        q.memcpy(usm_ptr, old_usm, sizeof(T) * val_count).wait();
    }

    shamlog_debug_alloc_ln("ResizableBufferUSM", events_hndl.get_hash_log(), "delete old buf");
    sycl::free(old_usm, q);
}

template<class T>
bool shamalgs::ResizableUSMBuffer<T>::check_buf_match(ResizableUSMBuffer<T> &f2) {

    bool match = true;

    match = match && (val_count == f2.val_count);

    {

        std::vector<sycl::event> wait_list;
        auto acc1 = get_usm_ptr_read_only(wait_list);
        auto acc2 = f2.get_usm_ptr_read_only(wait_list);

        sycl::buffer<u8> res_buf(val_count);

        sycl::event e = q.submit([&, acc1, acc2](sycl::handler &cgh) {
            cgh.depends_on(wait_list);

            sycl::accessor acc_res{res_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{val_count}, [=](sycl::item<1> i) {
                acc_res[i] = sham::equals(acc1[i], acc2[i]);
            });
        });

        register_read_event(e);
        f2.register_read_event(e);

        match = match && shamalgs::reduction::is_all_true(res_buf, f2.size());
    }

    return match;
}

#define X(a) template class shamalgs::ResizableUSMBuffer<a>;
XMAC_LIST_ENABLED_ResizableUSMBuffer
#undef X
