// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamrock/legacy/patch/utility/compute_field.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "interface_generator.hpp"
#include "shamrock/legacy/patch/comm/patchdata_exchanger.hpp"
#include <vector>

namespace impl {
    
    template <class vectype, class primtype>
    void comm_interfaces(PatchScheduler &sched, std::vector<InterfaceComm<vectype>> &interface_comm_list,
                        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<shamrock::patch::PatchData>>>> &interface_map,bool periodic) {
        StackEntry stack_loc{};
                            using namespace shamrock::patch;

        interface_map.clear();
        for (const Patch &p : sched.patch_list.global) {
            interface_map[p.id_patch] = std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>();
        }

        std::vector<std::unique_ptr<PatchData>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count > 0) {

                    auto patch_in = sched.patch_data.get_pdat(interface_comm_list[i].sender_patch_id);

                    std::vector<std::unique_ptr<PatchData>> pret = InterfaceVolumeGenerator::append_interface<vectype>(
                        shamsys::instance::get_alt_queue(), patch_in,
                        {interface_comm_list[i].interf_box_min}, {interface_comm_list[i].interf_box_max},interface_comm_list[i].interf_offset);
                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PatchData>(sched.pdl));
                }
                comm_vec.push_back(
                    u64_2{interface_comm_list[i].global_patch_idx_send, interface_comm_list[i].global_patch_idx_recv});
            }

            //std::cout << "\n split \n";
        }

        patch_data_exchange_object(sched.pdl,sched.patch_list.global, comm_pdat,comm_vec,interface_map);

    }




    template <class T,class vectype>
    void comm_interfaces_field(PatchScheduler &sched, PatchComputeField<T> &pcomp_field, std::vector<InterfaceComm<vectype>> &interface_comm_list,
                        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>> &interface_field_map,bool periodic) {

        StackEntry stack_loc{};
                            using namespace shamrock::patch;
        using PCField = PatchDataField<T>;

        interface_field_map.clear();
        for (const Patch &p : sched.patch_list.global) {
            interface_field_map[p.id_patch] = std::vector<std::tuple<u64, std::unique_ptr<PCField>>>();
        }

        std::vector<std::unique_ptr<PCField>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count > 0) {


                    std::vector<std::unique_ptr<PCField>> pret = InterfaceVolumeGenerator::append_interface_field<T,vectype>(
                        shamsys::instance::get_alt_queue(),
                        sched.patch_data.get_pdat(interface_comm_list[i].sender_patch_id),
                        pcomp_field.field_data.at(interface_comm_list[i].sender_patch_id),
                        {interface_comm_list[i].interf_box_min}, {interface_comm_list[i].interf_box_max});


                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PCField>("comp_field",1));
                }
                comm_vec.push_back(
                    u64_2{interface_comm_list[i].global_patch_idx_send, interface_comm_list[i].global_patch_idx_recv});
            }

            //std::cout << "\n split \n";
        }

        patch_data_field_exchange_object<T>(sched.patch_list.global, comm_pdat,comm_vec,interface_field_map);
        
    }

} // namespace impl