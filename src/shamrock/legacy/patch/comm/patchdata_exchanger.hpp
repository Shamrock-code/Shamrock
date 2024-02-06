// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file patchdata_exchanger.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "patchdata_exchanger_impl.hpp"
#include <vector>


[[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
inline void patch_data_exchange_object(
    shamrock::patch::PatchDataLayout & pdl,
    std::vector<shamrock::patch::Patch> & global_patch_list,
    std::vector<std::unique_ptr<shamrock::patch::PatchData>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<shamrock::patch::PatchData>>>> & interface_map
    ){
        patchdata_exchanger::impl::patch_data_exchange_object(pdl,global_patch_list, send_comm_pdat, send_comm_vec, interface_map);
    }


template<class T> 
[[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
inline void patch_data_field_exchange_object(
    std::vector<shamrock::patch::Patch> & global_patch_list,
    std::vector<std::unique_ptr<PatchDataField<T>>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>> & interface_map
    ){
        patchdata_exchanger::impl::patch_data_field_exchange_object(global_patch_list, send_comm_pdat, send_comm_vec, interface_map);
    }
