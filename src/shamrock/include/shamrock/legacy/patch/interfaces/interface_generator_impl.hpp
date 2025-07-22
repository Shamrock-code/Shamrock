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
 * @file interface_generator_impl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <vector>

namespace impl {

    // TODO make box list reference
    template<class vectype>
    std::vector<u8> get_flag_choice(
        sycl::queue &queue,
        shamrock::patch::PatchData &pdat,
        std::vector<vectype> boxs_min,
        std::vector<vectype> boxs_max);

    template<>
    inline std::vector<u8> get_flag_choice<f32_3>(
        sycl::queue &queue,
        shamrock::patch::PatchData &pdat,
        std::vector<f32_3> boxs_min,
        std::vector<f32_3> boxs_max) {

        if (boxs_min.size() > u8_max - 1) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "this algo is not build to handle more than 2^8 - 2 boxes as input");
        }

        using namespace shamrock::patch;

        // TODO change this func when implementing the USM patch without the pos_s_buf/pos_d_buf
        std::vector<u8> flag_choice(pdat.get_obj_cnt());

        if (!pdat.is_empty()) {

            sycl::buffer<u8> flag_buf(flag_choice.data(), flag_choice.size());

            sycl::buffer<f32_3> bmin_buf(boxs_min.data(), boxs_min.size());
            sycl::buffer<f32_3> bmax_buf(boxs_max.data(), boxs_max.size());

            sycl::range<1> range{pdat.get_obj_cnt()};

            u32 field_ipos = pdat.pdl.get_field_idx<f32_3>("xyz");

            PatchDataField<f32_3> &pos_field = pdat.get_field<f32_3>(field_ipos);

            auto sptr       = shamsys::instance::get_compute_scheduler_ptr();
            auto &q         = sptr->get_queue();
            auto &pos_s_buf = pos_field.get_buf();

            sham::EventList depends_list;
            const f32_3 *pos_s = pos_s_buf.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                auto bmin = bmin_buf.get_access<sycl::access::mode::read>(cgh);
                auto bmax = bmax_buf.get_access<sycl::access::mode::read>(cgh);

                auto index_box = flag_buf.get_access<sycl::access::mode::discard_write>(cgh);

                u8 num_boxes = boxs_min.size();

                cgh.parallel_for<class BuildInterfacef32>(range, [=](sycl::item<1> item) {
                    u64 i = (u64) item.get_id(0);

                    f32_3 pos_i  = pos_s[i];
                    index_box[i] = u8_max;

                    for (u8 idx = 0; idx < num_boxes; idx++) {

                        if (Patch::is_in_patch_converted(pos_i, bmin[idx], bmax[idx])) {
                            index_box[i] = idx;
                        }
                    }
                });
            });

            pos_s_buf.complete_event_state(e);
        }

        return flag_choice;
    }

    template<>
    inline std::vector<u8> get_flag_choice<f64_3>(
        sycl::queue &queue,
        shamrock::patch::PatchData &pdat,
        std::vector<f64_3> boxs_min,
        std::vector<f64_3> boxs_max) {

        if (boxs_min.size() > u8_max - 1) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "this algo is not build to handle more than 2^8 - 2 boxes as input");
        }

        using namespace shamrock::patch;

        std::vector<u8> flag_choice(pdat.get_obj_cnt());

        if (!pdat.is_empty()) {
            sycl::buffer<u8> flag_buf(flag_choice.data(), flag_choice.size());

            sycl::buffer<f64_3> bmin_buf(boxs_min.data(), boxs_min.size());
            sycl::buffer<f64_3> bmax_buf(boxs_max.data(), boxs_max.size());

            sycl::range<1> range{pdat.get_obj_cnt()};

            u32 field_ipos = pdat.pdl.get_field_idx<f64_3>("xyz");

            PatchDataField<f64_3> &pos_field = pdat.get_field<f64_3>(field_ipos);

            auto sptr       = shamsys::instance::get_compute_scheduler_ptr();
            auto &q         = sptr->get_queue();
            auto &pos_d_buf = pos_field.get_buf();

            sham::EventList depends_list;
            const f64_3 *pos_d = pos_d_buf.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                auto bmin = bmin_buf.get_access<sycl::access::mode::read>(cgh);
                auto bmax = bmax_buf.get_access<sycl::access::mode::read>(cgh);

                auto index_box = flag_buf.get_access<sycl::access::mode::discard_write>(cgh);

                u8 num_boxes = boxs_min.size();

                cgh.parallel_for<class BuildInterfacef64>(range, [=](sycl::item<1> item) {
                    u64 i = (u64) item.get_id(0);

                    f64_3 pos_i  = pos_d[i];
                    index_box[i] = u8_max;

                    for (u8 idx = 0; idx < num_boxes; idx++) {
                        if (Patch::is_in_patch_converted(pos_i, bmin[idx], bmax[idx])) {
                            index_box[i] = idx;
                        }
                    }
                });
            });

            pos_d_buf.complete_event_state(e);
        }

        return flag_choice;
    }

    template<class T, class vectype>
    inline std::vector<std::unique_ptr<PatchDataField<T>>> append_interface_field(
        sycl::queue &queue,
        shamrock::patch::PatchData &pdat,
        PatchDataField<T> &pdat_cfield,
        std::vector<vectype> boxs_min,
        std::vector<vectype> boxs_max) {

        std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

        std::vector<std::unique_ptr<PatchDataField<T>>> pdat_vec(boxs_min.size());
        for (auto &p : pdat_vec) {
            p = std::make_unique<PatchDataField<T>>("comp_field", 1);
        }

        std::vector<std::vector<u32>> idxs(boxs_min.size());

        for (u32 i = 0; i < flag_choice.size(); i++) {
            if (flag_choice[i] < boxs_min.size()) {
                idxs[flag_choice[i]].push_back(i);
            }
        }

        if (!pdat.is_empty()) {
            for (u32 i = 0; i < idxs.size(); i++) {
                pdat_cfield.append_subset_to(idxs[i], *pdat_vec[i]);
            }
        }

        return pdat_vec;
    }

} // namespace impl
