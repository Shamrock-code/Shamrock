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
 * @file compute_field.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <unordered_map>
#include <memory>
#include <vector>

template<class T>
class PatchComputeField {
    public:

    public:
    std::unordered_map<u64, PatchDataField<T>> field_data;

    inline void generate(PatchScheduler &sched) {

        using namespace shamrock::patch;
        // sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
        //     field_data.insert({id_patch,PatchDataField<T>("comp_field",1)});
        //     field_data.at(id_patch).resize(pdat_buf.element_count);
        // });

        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            field_data.insert({id_patch, PatchDataField<T>("comp_field", 1)});
            field_data.at(id_patch).resize(pdat.get_obj_cnt());
        });
    }

    inline void generate(PatchScheduler &sched, std::unordered_map<u64, u32> &size_map) {

        using namespace shamrock::patch;

        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            field_data.insert({id_patch, PatchDataField<T>("comp_field", 1)});
            field_data.at(id_patch).resize(size_map[id_patch]);
        });
    }

    private:
    // std::unordered_map<u64, std::unique_ptr<sycl::buffer<T>>> field_data_buf;

    public:
    inline const std::unique_ptr<sycl::buffer<T>> &get_buf(u64 id_patch) const {
        return field_data.at(id_patch).get_buf();
    }

    [[deprecated]]
    inline std::unique_ptr<sycl::buffer<T>> get_sub_buf(u64 id_patch) {
        return field_data.at(id_patch).get_sub_buf();
    }

    inline PatchDataField<T> &get_field(u64 id_patch) { return field_data.at(id_patch); }

    // inline void to_sycl(){
    //     for (auto & [key,dat] : field_data) {
    //         //field_data_buf[key] = dat.get_sub_buf();
    //     }
    // }
    //
    // inline void to_map(){
    //     //field_data_buf.clear();
    // }
};

template<class T>
class PatchComputeFieldInterfaces {
    public:
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>>
        interface_map;
};
