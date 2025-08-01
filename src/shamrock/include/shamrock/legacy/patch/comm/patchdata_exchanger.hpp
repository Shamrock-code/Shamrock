// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file patchdata_exchanger.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "patchdata_exchanger_impl.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include <vector>

[[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
inline void patch_data_exchange_object(
    shamrock::patch::PatchDataLayout &pdl,
    std::vector<shamrock::patch::Patch> &global_patch_list,
    std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<
        u64,
        std::vector<std::tuple<u64, std::unique_ptr<shamrock::patch::PatchDataLayer>>>> &interface_map) {
    patchdata_exchanger::impl::patch_data_exchange_object(
        pdl, global_patch_list, send_comm_pdat, send_comm_vec, interface_map);
}

template<class T>
[[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
inline void patch_data_field_exchange_object(
    std::vector<shamrock::patch::Patch> &global_patch_list,
    std::vector<std::unique_ptr<PatchDataField<T>>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>>
        &interface_map) {
    patchdata_exchanger::impl::patch_data_field_exchange_object(
        global_patch_list, send_comm_pdat, send_comm_vec, interface_map);
}
