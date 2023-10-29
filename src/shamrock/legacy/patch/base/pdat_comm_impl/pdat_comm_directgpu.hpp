// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pdat_comm_directgpu.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shamrock/legacy/patch/base/patchdata_field.hpp"

namespace impl::directgpu {

    using namespace patchdata_field;

    namespace send {
        template <class T> inline T *init(PatchDataField<T> &pdat_field, u32 comm_sz) {

            T *comm_ptr = sycl::malloc_device<T>(comm_sz, shamsys::instance::get_compute_queue());
            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_device", comm_sz, "->", comm_ptr);

            auto & buf = pdat_field.get_buf();

            if (pdat_field.size() > 0) {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                auto ker_copy = shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc{*buf, cgh, sycl::read_only};

                    T *ptr = comm_ptr;

                    cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) { ptr[item.get_linear_id()] = acc[item]; });
                });

                ker_copy.wait();
            } else {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
            }

            return comm_ptr;
        }

        template <class T> inline void finalize(T *comm_ptr) {
            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }
    } // namespace send

    namespace recv {
        template <class T> T *init(u32 comm_sz) {
            T *comm_ptr = sycl::malloc_device<T>(comm_sz, shamsys::instance::get_compute_queue());

            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_device", comm_sz);

            return comm_ptr;
        };

        template <class T> void finalize(PatchDataField<T> &pdat_field, T *comm_ptr, u32 comm_sz) {
            auto & buf = pdat_field.get_buf();

            if (pdat_field.size() > 0) {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                auto ker_copy = shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc{*buf, cgh, sycl::write_only};

                    T *ptr = comm_ptr;

                    cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) { acc[item] = ptr[item.get_linear_id()]; });
                });

                ker_copy.wait();
            } else {
                logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
            }

            logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }
    } // namespace recv




} // namespace impl::directgpu