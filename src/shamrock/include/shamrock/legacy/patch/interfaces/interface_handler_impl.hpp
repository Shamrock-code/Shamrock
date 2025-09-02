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
 * @file interface_handler_impl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "interface_generator.hpp"
#include "shamrock/legacy/patch/comm/patchdata_exchanger.hpp"
#include "shamrock/legacy/patch/utility/compute_field.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <vector>

namespace impl {

    template<class vectype, class primtype>
    [[deprecated("Legacy module")]]
    void comm_interfaces(
        PatchScheduler &sched,
        std::vector<InterfaceComm<vectype>> &interface_comm_list,
        std::unordered_map<
            u64,
            std::vector<std::tuple<u64, std::unique_ptr<shamrock::patch::PatchDataLayer>>>>
            &interface_map,
        bool periodic) {
        StackEntry stack_loc{};
        using namespace shamrock::patch;

        interface_map.clear();
        for (const Patch &p : sched.patch_list.global) {
            interface_map[p.id_patch]
                = std::vector<std::tuple<u64, std::unique_ptr<PatchDataLayer>>>();
        }

        std::vector<std::unique_ptr<PatchDataLayer>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].load_value
                    > 0) {

                    auto patch_in
                        = sched.patch_data.get_pdat(interface_comm_list[i].sender_patch_id);

                    std::vector<std::unique_ptr<PatchDataLayer>> pret
                        = InterfaceVolumeGenerator::append_interface<vectype>(
                            shamsys::instance::get_alt_queue(),
                            patch_in,
                            {interface_comm_list[i].interf_box_min},
                            {interface_comm_list[i].interf_box_max},
                            interface_comm_list[i].interf_offset);
                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PatchDataLayer>(sched.get_layout_ptr()));
                }
                comm_vec.push_back(
                    u64_2{
                        interface_comm_list[i].global_patch_idx_send,
                        interface_comm_list[i].global_patch_idx_recv});
            }

            // std::cout << "\n split \n";
        }

        patch_data_exchange_object(
            sched.get_layout_ptr(), sched.patch_list.global, comm_pdat, comm_vec, interface_map);
    }

    template<class T, class vectype>
    [[deprecated("Legacy module")]]
    void comm_interfaces_field(
        PatchScheduler &sched,
        PatchComputeField<T> &pcomp_field,
        std::vector<InterfaceComm<vectype>> &interface_comm_list,
        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>>
            &interface_field_map,
        bool periodic) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;
        using PCField = PatchDataField<T>;

        interface_field_map.clear();
        for (const Patch &p : sched.patch_list.global) {
            interface_field_map[p.id_patch]
                = std::vector<std::tuple<u64, std::unique_ptr<PCField>>>();
        }

        std::vector<std::unique_ptr<PCField>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].load_value
                    > 0) {

                    std::vector<std::unique_ptr<PCField>> pret
                        = InterfaceVolumeGenerator::append_interface_field<T, vectype>(
                            shamsys::instance::get_alt_queue(),
                            sched.patch_data.get_pdat(interface_comm_list[i].sender_patch_id),
                            pcomp_field.field_data.at(interface_comm_list[i].sender_patch_id),
                            {interface_comm_list[i].interf_box_min},
                            {interface_comm_list[i].interf_box_max});

                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PCField>("comp_field", 1));
                }
                comm_vec.push_back(
                    u64_2{
                        interface_comm_list[i].global_patch_idx_send,
                        interface_comm_list[i].global_patch_idx_recv});
            }

            // std::cout << "\n split \n";
        }

        patch_data_field_exchange_object<T>(
            sched.patch_list.global, comm_pdat, comm_vec, interface_field_map);
    }

} // namespace impl
