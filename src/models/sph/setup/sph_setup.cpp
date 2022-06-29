#include "sph_setup.hpp"
#include "models/sph/base/kernels.hpp"
#include "core/patch/comm/patch_object_mover.hpp"

#include "core/sys/mpi_handler.hpp"


template<class flt, class u_morton, class Kernel>
void models::sph::SetupSPH<flt,u_morton,Kernel>::init(PatchScheduler & sched){
    if (mpi_handler::world_rank == 0) {
        Patch root;

        root.node_owner_id = mpi_handler::world_rank;

        root.x_min = 0;
        root.y_min = 0;
        root.z_min = 0;

        root.x_max = HilbertLB::max_box_sz;
        root.y_max = HilbertLB::max_box_sz;
        root.z_max = HilbertLB::max_box_sz;

        root.pack_node_index = u64_max;

        PatchData pdat(sched.pdl);

        root.data_count = pdat.get_obj_cnt();
        root.load_value = pdat.get_obj_cnt();

        sched.add_patch(root,pdat);  

    } else {
        sched.patch_list._next_patch_id++;
    }  

    mpi::barrier(MPI_COMM_WORLD);

    sched.owned_patch_id = sched.patch_list.build_local();

    sched.patch_list.build_global();

    sched.patch_tree.build_from_patchtable(sched.patch_list.global, HilbertLB::max_box_sz);

    std::cout << "build local" << std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();
}


template<class flt, class u_morton, class Kernel>
void models::sph::SetupSPH<flt,u_morton,Kernel>::add_particules_fcc(PatchScheduler & sched, flt dr, std::tuple<vec,vec> box){

    if(mpi_handler::world_rank == 0){
        std::vector<vec> vec_acc;

        add_particles_fcc(
            dr, 
            box , 
            [](sycl::vec<flt,3> r){return true;}, 
            [&](sycl::vec<flt,3> r,flt h){
                vec_acc.push_back(r); 
            });

        std::cout << ">>> adding : " << vec_acc.size() << " objects" << std::endl;

        PatchData tmp(sched.pdl);
        tmp.resize(vec_acc.size());

        part_cnt += vec_acc.size();

        PatchDataField<vec> & f = tmp.get_field<vec>(sched.pdl.get_field_idx<vec>("xyz"));

        sycl::buffer<vec> buf (vec_acc.data(),vec_acc.size());

        f.override(buf);

        if(sched.owned_patch_id.empty()) throw shamrock_exc("the scheduler does not have patch in that rank");

        u64 insert_id = *sched.owned_patch_id.begin();

        sched.patch_data.owned_data.at(insert_id).insert_elements(tmp);
    }


    //TODO apply position modulo here

    sched.scheduler_step(false, false);

    {
        SerialPatchTree<vec> sptree(sched.patch_tree, sched.get_box_tranform<vec>());
        sptree.attach_buf();
        reatribute_particles(sched, sptree, periodic_mode);
    }

    sched.scheduler_step(true, true);

    for (auto & [pid,pdat] : sched.patch_data.owned_data) {

        PatchDataField<flt> & f = pdat.template get_field<flt>(sched.pdl.get_field_idx<flt>("hpart"));

        f.override(dr);
    }
}

template class models::sph::SetupSPH<f32,u32,models::sph::kernels::M4<f32>>;
