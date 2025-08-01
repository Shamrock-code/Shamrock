// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file merged_patch.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/legacy/patch/utility/merged_patch.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"

template<class flt>
auto MergedPatchData<flt>::merge_patches(
    PatchScheduler &sched, LegacyInterfacehandler<vec, flt> &interface_hndl)
    -> std::unordered_map<u64, MergedPatchData<flt>> {

    using namespace shamrock::patch;

    std::unordered_map<u64, MergedPatchData<flt>> merged_data;

    sched.for_each_patch_data([&](u64 id_patch, Patch &p, PatchDataLayer &pdat) {
        merged_data.emplace(id_patch, sched.pdl);

        auto pbox            = sched.patch_data.sim_box.get_box<flt>(p);
        u32 original_element = pdat.get_obj_cnt();

        MergedPatchData<flt> &ret = merged_data.at(id_patch);

        ret.data.insert_elements(pdat);

        interface_hndl.for_each_interface(
            id_patch,
            [&](u64 patch_id,
                u64 interf_patch_id,
                PatchDataLayer &interfpdat,
                std::tuple<vec, vec> box) {
                std::get<0>(pbox) = sycl::min(std::get<0>(box), std::get<0>(pbox));
                std::get<1>(pbox) = sycl::max(std::get<1>(box), std::get<1>(pbox));

                ret.data.insert_elements(interfpdat);
            });

        ret.box            = pbox;
        ret.or_element_cnt = original_element;
    });

    return merged_data;
}

template<class flt, class T>
auto MergedPatchCompField<flt, T>::merge_patches_cfield(
    PatchScheduler &sched,
    LegacyInterfacehandler<vec, flt> &interface_hndl,
    PatchComputeField<T> &comp_field,
    PatchComputeFieldInterfaces<T> &comp_field_interf)
    -> std::unordered_map<u64, MergedPatchCompField<flt, T>> {

    using namespace shamrock::patch;

    std::unordered_map<u64, MergedPatchCompField<flt, T>> merged_data;

    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
        auto &compfield = comp_field.get_field(id_patch);

        merged_data.insert({id_patch, MergedPatchCompField<flt, T>()});

        auto &merged_field = merged_data.at(id_patch);

        merged_field.or_element_cnt = compfield.get_val_cnt();
        merged_field.buf.insert(compfield);

        std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>> &p_interf_lst
            = comp_field_interf.interface_map[id_patch];

        for (auto &[int_pid, pdat_ptr] : p_interf_lst) {

            merged_field.buf.insert(*pdat_ptr);
        }
    });

    return merged_data;
}

template class MergedPatchData<f32>;
template class MergedPatchData<f64>;

#define X(arg)                                                                                     \
    template class MergedPatchCompField<f32, arg>;                                                 \
    template class MergedPatchCompField<f64, arg>;
XMAC_LIST_ENABLED_FIELD
#undef X
