// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file interface_generator.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/patch/interfaces/interface_generator.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/Patch.hpp"
#include <memory>
#include <stdexcept>
#include <vector>
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/legacy/patch/interfaces/interface_generator_impl.hpp"
#include "shamrock/scheduler/SchedulerPatchData.hpp"
#include "shamtree/kernels/geometry_utils.hpp"

// TODO can merge those 2 func

template<>
std::vector<std::unique_ptr<shamrock::patch::PatchData>>
InterfaceVolumeGenerator::append_interface<f32_3>(
    sycl::queue &queue,
    shamrock::patch::PatchData &pdat,
    std::vector<f32_3> boxs_min,
    std::vector<f32_3> boxs_max,
    f32_3 add_offset) {

    using namespace shamrock::patch;

    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto &p : pdat_vec) {
        p = std::make_unique<PatchData>(pdat.pdl);
    }

    std::vector<std::vector<u32>> idxs(boxs_min.size());

    for (u32 i = 0; i < flag_choice.size(); i++) {
        if (flag_choice[i] < boxs_min.size()) {
            idxs[flag_choice[i]].push_back(i);
        }
    }

    if (!pdat.is_empty()) {
        for (u32 i = 0; i < idxs.size(); i++) {
            pdat.append_subset_to(idxs[i], *pdat_vec[i]);
            u32 ixyz = pdat.pdl.get_field_idx<f32_3>("xyz");
            pdat_vec[i]->get_field<f32_3>(ixyz).apply_offset(add_offset);
        }
    }

    return pdat_vec;
}

template<>
std::vector<std::unique_ptr<shamrock::patch::PatchData>>
InterfaceVolumeGenerator::append_interface<f64_3>(
    sycl::queue &queue,
    shamrock::patch::PatchData &pdat,
    std::vector<f64_3> boxs_min,
    std::vector<f64_3> boxs_max,
    f64_3 add_offset) {
    using namespace shamrock::patch;

    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto &p : pdat_vec) {
        p = std::make_unique<PatchData>(pdat.pdl);
    }

    std::vector<std::vector<u32>> idxs(boxs_min.size());

    for (u32 i = 0; i < flag_choice.size(); i++) {
        if (flag_choice[i] < boxs_min.size()) {
            idxs[flag_choice[i]].push_back(i);
        }
    }

    if (!pdat.is_empty()) {
        for (u32 i = 0; i < idxs.size(); i++) {
            pdat.append_subset_to(idxs[i], *pdat_vec[i]);
            u32 ixyz = pdat.pdl.get_field_idx<f64_3>("xyz");
            pdat_vec[i]->get_field<f64_3>(ixyz).apply_offset(add_offset);
        }
    }

    return pdat_vec;
}
