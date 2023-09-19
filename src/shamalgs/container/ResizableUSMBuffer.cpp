// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "ResizableUSMBuffer.hpp"

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

#define X(a) template class shamalgs::ResizableUSMBuffer<a>;
XMAC_LIST_ENABLED_ResizableUSMBuffer
#undef X