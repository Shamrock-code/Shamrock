// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "ResizableUSMBuffer.hpp"
#include "shamalgs/details/reduction/reduction.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"

template<class T>
void shamalgs::ResizableUSMBuffer<T>::alloc() {
    if (usm_ptr != nullptr) {
        throw shambase::throw_with_loc<std::runtime_error>("the usm pointer is already allocated");
    }

    if (type == Host) {
        usm_ptr = sycl::malloc_host<T>(capacity, q);
    } else if (type == Device) {
        usm_ptr = sycl::malloc_device<T>(capacity, q);
    } else if (type == Shared) {
        usm_ptr = sycl::malloc_shared<T>(capacity, q);
    }
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::free() {
    sycl::free(usm_ptr, q);
    usm_ptr = nullptr;
}

template<class T>
void shamalgs::ResizableUSMBuffer<T>::change_capacity(u32 new_capa) {
    StackEntry stack_loc{false};

    logger::debug_alloc_ln(
        "ResizableBuffer", "change capacity from : ", capacity, "to :", new_capa);

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

                logger::debug_alloc_ln("PatchDataField", "delete old buf : ");
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

    logger::debug_alloc_ln("ResizableBuffer", "resize from : ", val_count, "to :", new_size);

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

    logger::debug_alloc_ln("PatchDataField", "delete old buf : ");
    sycl::free(old_usm, q);
}

template<class T>
shamalgs::ResizableUSMBuffer<T>::ResizableUSMBuffer::~ResizableUSMBuffer() {
    if (usm_ptr != nullptr) {
        free();
    }
}




template<class T>
bool shamalgs::ResizableUSMBuffer<T>::check_buf_match(ResizableUSMBuffer<T> &f2){

    bool match = true;

    match = match && (val_count == f2.val_count);

    {

        std::vector<sycl::event> wait_list;
        auto acc1 = get_usm_ptr_read_only(wait_list);
        auto acc2 = f2.get_usm_ptr_read_only(wait_list);

        sycl::buffer<u8> res_buf(val_count);

        sycl::event e = shamsys::instance::get_compute_queue().submit([&,acc1,acc2](sycl::handler &cgh) {
            cgh.depends_on(wait_list);

            sycl::accessor acc_res{res_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{val_count}, [=](sycl::item<1> i) {
                acc_res[i] = shambase::vec_equals(acc1[i], acc2[i]);
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