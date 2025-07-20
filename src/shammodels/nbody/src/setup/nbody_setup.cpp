// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file nbody_setup.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/nbody/setup/nbody_setup.hpp"
#include "shamrock/legacy/patch/comm/patch_object_mover.hpp"
#include "shamrock/patch/Patch.hpp"

template<class flt>
void models::nbody::NBodySetup<flt>::init(PatchScheduler &sched) {

    using namespace shamrock::patch;

    sched.add_root_patch();

    std::cout << "build local" << std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](shamrock::patch::Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
}

template<class flt>
void models::nbody::NBodySetup<flt>::add_particules_fcc(
    PatchScheduler &sched, flt dr, std::tuple<vec, vec> box) {

    using namespace shamrock::patch;

    if (shamcomm::world_rank() == 0) {
        std::vector<vec> vec_acc;

        generic::setup::generators::add_particles_fcc(
            dr,
            box,
            [&box](sycl::vec<flt, 3> r) {
                return BBAA::is_coord_in_range(r, std::get<0>(box), std::get<1>(box));
            },
            [&](sycl::vec<flt, 3> r, flt h) {
                vec_acc.push_back(r);
            });

        std::cout << ">>> adding : " << vec_acc.size() << " objects" << std::endl;

        PatchData tmp(sched.pdl);
        tmp.resize(vec_acc.size());

        part_cnt += vec_acc.size();

        {
            u32 len                = vec_acc.size();
            PatchDataField<vec> &f = tmp.get_field<vec>(sched.pdl.get_field_idx<vec>("xyz"));
            sycl::buffer<vec> buf(vec_acc.data(), len);
            f.override(buf, len);
        }

        if (sched.owned_patch_id.empty())
            throw shambase::make_except_with_loc<std::runtime_error>(
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

        SerialPatchTree<vec> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<vec>());
        sptree.attach_buf();
        reatribute_particles(sched, sptree, periodic_mode);
    }

    sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData &pdat) {
        std::cout << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << std::endl;
    });

    // std::cout << sched.dump_status() << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n"<< std::endl;

    sched.scheduler_step(true, false);

    // std::cout << sched.dump_status() << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n"<< std::endl;

    sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData &pdat) {
        std::cout << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << std::endl;
    });
}

template class models::nbody::NBodySetup<f32>;
