// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pdat_comm_cp_to_host.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamrock/legacy/patch/base/patchdata_field.hpp"

namespace impl::copy_to_host {

    using namespace patchdata_field;

    namespace send {
        template<class T>
        inline T *init(PatchDataField<T> &pdat_field, u32 comm_sz) {

            T *comm_ptr = sycl::malloc_host<T>(comm_sz, shamsys::instance::get_compute_queue());
            shamlog_debug_sycl_ln(
                "PatchDataField MPI Comm", "sycl::malloc_host", comm_sz, "->", comm_ptr);

            auto &buf = pdat_field.get_buf();

            if (pdat_field.size() > 0) {
                shamlog_debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                auto ker_copy
                    = shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                          sycl::accessor acc{*buf, cgh, sycl::read_only};

                          T *ptr = comm_ptr;

                          cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) {
                              ptr[item.get_linear_id()] = acc[item];
                          });
                      });

                ker_copy.wait();
            } else {
                shamlog_debug_sycl_ln(
                    "PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
            }

            return comm_ptr;
        }

        template<class T>
        inline void finalize(T *comm_ptr) {
            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }
    } // namespace send

    namespace recv {
        template<class T>
        T *init(u32 comm_sz) {
            T *comm_ptr = sycl::malloc_host<T>(comm_sz, shamsys::instance::get_compute_queue());

            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host", comm_sz);

            return comm_ptr;
        };

        template<class T>
        void finalize(PatchDataField<T> &pdat_field, T *comm_ptr, u32 comm_sz) {
            auto &buf = pdat_field.get_buf();

            if (pdat_field.size() > 0) {
                shamlog_debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                auto ker_copy
                    = shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                          sycl::accessor acc{*buf, cgh, sycl::write_only};

                          T *ptr = comm_ptr;

                          cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) {
                              acc[item] = ptr[item.get_linear_id()];
                          });
                      });

                ker_copy.wait();
            } else {
                shamlog_debug_sycl_ln(
                    "PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
            }

            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }
    } // namespace recv

} // namespace impl::copy_to_host
