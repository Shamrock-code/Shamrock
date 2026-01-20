// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/legacy/io/dump.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/interfaces/interface_generator.hpp"
#include "shamrock/legacy/patch/interfaces/interface_handler.hpp"
#include "shamrock/legacy/patch/interfaces/interface_selector.hpp"
#include "shamrock/legacy/patch/utility/patch_field.hpp"
#include "shamrock/legacy/patch/utility/patch_reduc_tree.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtest/shamtest.hpp"
#include <string>

class Reduce_DataCount {
    public:
    static u64 reduce(u64 v0, u64 v1, u64 v2, u64 v3, u64 v4, u64 v5, u64 v6, u64 v7) {
        return v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
};

#if false
Test_start("patch::patch_reduc_tree::", generation, -1) {



    SchedulerMPI sched = SchedulerMPI(2500, 1);
    sched.init_mpi_required_types();

    patchdata_layout::set(1, 0, 0, 0, 0, 0);
    patchdata_layout::sync(MPI_COMM_WORLD);

    if (shamcomm::world_rank() == 0) {
        Patch p;

        p.data_count    = 1e4;
        p.load_value    = 1e4;
        p.node_owner_id = shamcomm::world_rank();

        p.x_min = 0;
        p.y_min = 0;
        p.z_min = 0;

        p.x_max = HilbertLB::max_box_sz;
        p.y_max = HilbertLB::max_box_sz;
        p.z_max = HilbertLB::max_box_sz;

        p.pack_node_index = u64_max;

        PatchData pdat;

        std::mt19937 eng(0x1111);
        std::uniform_real_distribution<f32> distpos(-1, 1);

        for (u32 part_id = 0; part_id < p.data_count; part_id++)
            pdat.pos_s.push_back({distpos(eng), distpos(eng), distpos(eng)});

        sched.add_patch(p, pdat);

    } else {
        sched.patch_list._next_patch_id++;
    }

    sched.owned_patch_id = sched.patch_list.build_local();

    // std::cout << sched.dump_status() << std::endl;
    sched.patch_list.build_global();
    // std::cout << sched.dump_status() << std::endl;

    //*
    sched.patch_tree.build_from_patchtable(sched.patch_list.global, HilbertLB::max_box_sz);
    sched.patch_data.sim_box.min_box_sim_s = {-1};
    sched.patch_data.sim_box.max_box_sim_s = {1};

    // std::cout << sched.dump_status() << std::endl;

    std::cout << "build local" << std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();

    // sched.patch_list.build_global();

    {
        PatchField<u64> dtcnt_field;
        dtcnt_field.local_nodes_value.resize(sched.patch_list.local.size());
        for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
            dtcnt_field.local_nodes_value[idx] = sched.patch_list.local[idx].data_count;
        }

        std::cout << "dtcnt_field.build_global(mpi_type_u64);" << std::endl;
        dtcnt_field.build_global(mpi_type_u64);

        std::cout << "len 1 : " << dtcnt_field.local_nodes_value.size() << std::endl;
        std::cout << "len 2 : " << dtcnt_field.global_values.size() << std::endl;

        SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());

        std::cout << "len 3 : " << sptree.get_element_count() << std::endl;

        std::cout << "sptree.attach_buf();" << std::endl;
        sptree.attach_buf();

        std::cout << "sptree.reduce_field" << std::endl;
        PatchFieldReduction<u64> pfield_reduced =
            sptree.reduce_field<u64, Reduce_DataCount>(shamsys::instance::get_alt_queue(), sched, dtcnt_field);

        std::cout << "pfield_reduced.detach_buf()" << std::endl;
        pfield_reduced.detach_buf();
        std::cout << " ------ > " << pfield_reduced.tree_field[0] << "\n\n\n";

        PatchField<f32> h_field;
        h_field.local_nodes_value.resize(sched.patch_list.local.size());
        for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
            h_field.local_nodes_value[idx] = 0.1f;
        }
        h_field.build_global(mpi_type_f32);

        InterfaceHandler<f32_3, f32> interface_hndl;
        interface_hndl.compute_interface_list<InterfaceSelector_SPH<f32_3, f32>>(sched, sptree, h_field,false);
        interface_hndl.comm_interfaces(sched,false);
        interface_hndl.print_current_interf_map();



        sched.dump_local_patches(format("patches_%d_node%d", 0, shamcomm::world_rank()));
    }

    for (u32 stepi = 1; stepi < 6; stepi++) {
        std::cout << " ------ step time = " << stepi << " ------" << std::endl;
        // std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        std::cout << " reduc " << std::endl;
        {

            // std::cout << sched.dump_status() << std::endl;

            PatchField<u64> dtcnt_field;
            dtcnt_field.local_nodes_value.resize(sched.patch_list.local.size());
            for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
                dtcnt_field.local_nodes_value[idx] = sched.patch_list.local[idx].data_count;
            }

            std::cout << "dtcnt_field.build_global(mpi_type_u64);" << std::endl;
            dtcnt_field.build_global(mpi_type_u64);

            // std::cout << "len 1 : " << dtcnt_field.local_nodes_value.size() << std::endl;
            // std::cout << "len 2 : " << dtcnt_field.global_values.size() << std::endl;

            SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());
            // sptree.dump_dat();

            // std::cout << "len 3 : " << sptree.get_element_count() << std::endl;

            // std::cout << "sptree.attach_buf();" << std::endl;
            sptree.attach_buf();

            // std::cout << "sptree.reduce_field" << std::endl;
            PatchFieldReduction<u64> pfield_reduced =
                sptree.reduce_field<u64, Reduce_DataCount>(shamsys::instance::get_alt_queue(), sched, dtcnt_field);

            // std::cout << "pfield_reduced.detach_buf()" << std::endl;
            pfield_reduced.detach_buf();
            std::cout << " ------ > " << pfield_reduced.tree_field[0] << "\n\n\n";

            PatchField<f32> h_field;
            h_field.local_nodes_value.resize(sched.patch_list.local.size());
            for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
                h_field.local_nodes_value[idx] = 0.1f;
            }
            h_field.build_global(mpi_type_f32);

            InterfaceHandler<f32_3, f32> interface_hndl;
            interface_hndl.compute_interface_list<InterfaceSelector_SPH<f32_3, f32>>(sched, sptree, h_field,false);
            interface_hndl.comm_interfaces(sched,false);
            interface_hndl.print_current_interf_map();

            sched.dump_local_patches(format("patches_%d_node%d", stepi, shamcomm::world_rank()));


            for(auto & [id,pdat] : sched.patch_data.owned_data){
                if(pdat.pos_s.size() > 0){
                    sycl::buffer<f32_3> pos(pdat.pos_s.data(),pdat.pos_s.size());

                    sycl::buffer<u64> newid = __compute_object_patch_owner<f32_3, class ComputeObejctPatchOwners>(
                        shamsys::instance::get_compute_queue(),
                        pos,
                        sptree);

                    {
                        auto nid = newid.get_access<sycl::access::mode::read>();

                        for(u32 i = 0 ; i < pdat.pos_s.size() ; i++){
                            std::cout <<id  << " " << i << " " <<nid[i] << "\n";
                        }std::cout << std::endl;
                    }
                }
            }

            dump_state("step"+std::to_string(stepi)+"/",sched,0);
        }

        // TODO test if a interface of size 0.5x0.5x0.5 exist == error
    }

    // std::cout << sched.dump_status() << std::endl;

    std::cout << "changing crit\n";
    sched.crit_patch_merge = 30;
    sched.crit_patch_split = 100;
    sched.scheduler_step(true, true);

    // std::cout << sched.dump_status() << std::endl;
    //*/

    sched.free_mpi_required_types();
}
#endif
