// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good

#include "sph_setup.hpp"
#include "shamrock/legacy/patch/comm/patch_object_mover.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/sph/kernels.hpp"

template<class flt, class Kernel>
void models::sph::SetupSPH<flt, Kernel>::init(PatchScheduler &sched) {
    StackEntry stack_loc{};
    using namespace shamrock::patch;

    sched.add_root_patch();

    std::cout << "build local" << std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();
}

template<class flt, class Kernel>
void models::sph::SetupSPH<flt, Kernel>::add_particules_fcc(PatchScheduler &sched,
                                                            flt dr,
                                                            std::tuple<vec, vec> box) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    if (shamsys::instance::world_rank == 0) {
        std::vector<vec> vec_acc;

        generic::setup::generators::add_particles_fcc(
            dr,
            box,
            [&box](sycl::vec<flt, 3> r) {
                return BBAA::is_coord_in_range(r, std::get<0>(box), std::get<1>(box));
            },
            [&](sycl::vec<flt, 3> r, flt h) { vec_acc.push_back(r); });

        std::cout << ">>> adding : " << vec_acc.size() << " objects" << std::endl;

        PatchData tmp(sched.pdl);
        tmp.resize(vec_acc.size());
        tmp.fields_raz();

        part_cnt += vec_acc.size();

        {
            u32 len                = vec_acc.size();
            PatchDataField<vec> &f = tmp.get_field<vec>(sched.pdl.get_field_idx<vec>("xyz"));
            sycl::buffer<vec> buf(vec_acc.data(), len);
            f.override(buf, len);
        }

        {
            PatchDataField<flt> &f = tmp.get_field<flt>(sched.pdl.get_field_idx<flt>("hpart"));
            f.override(dr);
        }

        if (sched.owned_patch_id.empty())
            throw shambase::throw_with_loc<std::runtime_error>(
                "the scheduler does not have patch in that rank");

        u64 insert_id = *sched.owned_patch_id.begin();

        sched.patch_data.get_pdat(insert_id).insert_elements(tmp);
    }

    // TODO apply position modulo here

    sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData &pdat) {
        std::cout << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << std::endl;
    });

    sched.scheduler_step(false, false);

    sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData &pdat) {
        std::cout << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << std::endl;
    });

    {
        auto [m, M] = sched.get_box_tranform<vec>();

        std::cout << "box transf" << m.x() << " " << m.y() << " " << m.z() << " | " << M.x() << " "
                  << M.y() << " " << M.z() << std::endl;

        SerialPatchTree<vec> sptree(sched.patch_tree, sched.get_sim_box().get_patch_transform<vec>());

        //sptree.print_status();


        shamrock::ReattributeDataUtility reatrib(sched);

        sptree.attach_buf();
        //reatribute_particles(sched, sptree, periodic_mode);

        reatrib.reatribute_patch_objects(sptree, "xyz");
    }

    sched.check_patchdata_locality_corectness();

    sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData &pdat) {
        std::cout << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << std::endl;
    });

    sched.scheduler_step(true, true);

    sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData &pdat) {
        std::cout << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << std::endl;
    });
}


template<class flt, class Kernel>
auto models::sph::SetupSPH<flt, Kernel>::get_closest_part_to(PatchScheduler &sched,vec pos) -> vec{

    using namespace shamrock::patch;

    vec best_dr = shambase::VectorProperties<vec>::get_max();
    flt best_dist2 = shambase::VectorProperties<flt>::get_max();

    sched.for_each_patchdata_nonempty([&](const Patch, PatchData & pdat){
        sycl::buffer<vec> & xyz = shambase::get_check_ref(pdat.get_field<vec>(0).get_buf());

        sycl::host_accessor acc {xyz, sycl::read_only};

        u32 cnt = pdat.get_obj_cnt();

        for(u32 i = 0; i < cnt; i++){
            vec tmp = acc[i];
            vec dr = tmp - pos;
            flt dist2 = sycl::dot(dr,dr);
            if(dist2 < best_dist2){
                best_dr = dr;
                best_dist2 = dist2;
            }
        }
    });


    std::vector<vec> list_dr {};
    shamalgs::collective::vector_allgatherv(std::vector<vec>{best_dr},list_dr,MPI_COMM_WORLD);


    for(vec tmp : list_dr){
        vec dr = tmp - pos;
        flt dist2 = sycl::dot(dr,dr);
        if(dist2 < best_dist2){
            best_dr = dr;
            best_dist2 = dist2;
        }
    }

    return pos + best_dr;

}




template class models::sph::SetupSPH<f32, shamrock::sph::kernels::M4<f32>>;
template class models::sph::SetupSPH<f32, shamrock::sph::kernels::M6<f32>>;
template class models::sph::SetupSPH<f64, shamrock::sph::kernels::M4<f64>>;
template class models::sph::SetupSPH<f64, shamrock::sph::kernels::M6<f64>>;
