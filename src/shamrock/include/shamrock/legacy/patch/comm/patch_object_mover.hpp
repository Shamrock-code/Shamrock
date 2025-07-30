// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file patch_object_mover.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "patchdata_exchanger.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include <unordered_map>

template<class vecprec>
[[deprecated("Legacy module")]]
inline std::unordered_map<u64, sycl::buffer<u64>>
get_new_id_map(PatchScheduler &sched, SerialPatchTree<vecprec> &sptree);

template<>
[[deprecated("Legacy module")]]
inline std::unordered_map<u64, sycl::buffer<u64>>
get_new_id_map<f32_3>(PatchScheduler &sched, SerialPatchTree<f32_3> &sptree) {

    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;

    sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
        if (!pdat.is_empty()) {

            u32 ixyz                         = sched.pdl.get_field_idx<f32_3>("xyz");
            PatchDataField<f32_3> &xyz_field = pdat.get_field<f32_3>(ixyz);

            if (xyz_field.get_nvar() != 1) {
                shambase::throw_unimplemented();
            }

            auto &pos = xyz_field.get_buf();

            newid_buf_map.insert(
                {id,
                 sptree.compute_patch_owner(
                     shamsys::instance::get_compute_scheduler_ptr(),
                     pos,
                     xyz_field.get_obj_cnt())});
        }
    });

    return newid_buf_map;
}

template<>
[[deprecated("Legacy module")]]
inline std::unordered_map<u64, sycl::buffer<u64>>
get_new_id_map<f64_3>(PatchScheduler &sched, SerialPatchTree<f64_3> &sptree) {

    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;

    sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
        if (!pdat.is_empty()) {

            u32 ixyz                         = sched.pdl.get_field_idx<f64_3>("xyz");
            PatchDataField<f64_3> &xyz_field = pdat.get_field<f64_3>(ixyz);

            if (xyz_field.get_nvar() != 1) {
                shambase::throw_unimplemented();
            }

            auto &pos = xyz_field.get_buf();

            newid_buf_map.insert(
                {id,
                 sptree.compute_patch_owner(
                     shamsys::instance::get_compute_scheduler_ptr(),
                     pos,
                     xyz_field.get_obj_cnt())});
        }
    });

    return newid_buf_map;
}

template<class vecprec>
[[deprecated("Legacy module")]]
inline void
reatribute_particles(PatchScheduler &sched, SerialPatchTree<vecprec> &sptree, bool periodic);

template<>
[[deprecated("Legacy module")]]
inline void
reatribute_particles<f32_3>(PatchScheduler &sched, SerialPatchTree<f32_3> &sptree, bool periodic) {

    using namespace shamrock::patch;

    bool err_id_in_newid = false;
    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;
    sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
        if (!pdat.is_empty()) {

            u32 ixyz                         = sched.pdl.get_field_idx<f32_3>("xyz");
            PatchDataField<f32_3> &xyz_field = pdat.get_field<f32_3>(ixyz);

            if (xyz_field.get_nvar() != 1) {
                shambase::throw_unimplemented();
            }

            auto &pos = xyz_field.get_buf();

            newid_buf_map.insert(
                {id,
                 sptree.compute_patch_owner(
                     shamsys::instance::get_compute_scheduler_ptr(),
                     pos,
                     xyz_field.get_obj_cnt())});

            {
                // auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
                sycl::host_accessor nid{newid_buf_map.at(id), sycl::read_only};
                for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                    bool err        = nid[i] == u64_max;
                    err_id_in_newid = err_id_in_newid || (err);
                }
            }

            if (periodic && err_id_in_newid) {
                // auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
                sycl::host_accessor nid{newid_buf_map.at(id), sycl::read_only};
                for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                    bool err = nid[i] == u64_max;

                    if (periodic && err) {
                        logger::err_ln(
                            "Patch Object Mover", "id = ", i, "is out of bound with periodic mode");
                        throw shambase::make_except_with_loc<std::runtime_error>("error");
                    }
                }
            }
        }
    });

    shamlog_debug_ln("Patch Object Mover", "err_id_in_newid :", err_id_in_newid);

    bool synced_should_res_box = sched.should_resize_box(err_id_in_newid);

    if (periodic && synced_should_res_box) {

        throw shambase::make_except_with_loc<std::runtime_error>(
            "box cannot be resized in periodic mode");
    }

    if (synced_should_res_box) {
        sched.patch_data.sim_box.reset_box_size();

        auto [bmin_, bmax_] = sched.patch_data.sim_box.get_bounding_box<f32_3>();

        f32_3 bmin = bmin_;
        f32_3 bmax = bmax_;

        sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
            u32 ixyz                         = sched.pdl.get_field_idx<f32_3>("xyz");
            PatchDataField<f32_3> &xyz_field = pdat.get_field<f32_3>(ixyz);

            {

                auto &pos = xyz_field.get_buf();
                auto acc  = pos.copy_to_stdvec();

                for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {

                    f32_3 r = acc[i];
                    bmin    = sycl::min(bmin, r);
                    bmax    = sycl::max(bmax, r);
                }
            }
        });

        sched.patch_data.sim_box.allreduce_set_bounding_box<f32_3>({bmin, bmax});
        sched.patch_data.sim_box.clean_box<f32>(1.2);

        shamlog_debug_ln(
            "Patch Object Mover",
            "resize box to  :",
            sched.patch_data.sim_box.get_bounding_box<f32_3>());

        sptree.detach_buf();
        sptree = SerialPatchTree<f32_3>(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<f32_3>());
        sptree.attach_buf();

        sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
            if (!pdat.is_empty()) {
                u32 ixyz                         = sched.pdl.get_field_idx<f32_3>("xyz");
                PatchDataField<f32_3> &xyz_field = pdat.get_field<f32_3>(ixyz);

                if (xyz_field.get_nvar() != 1) {
                    shambase::throw_unimplemented();
                }

                auto &pos = xyz_field.get_buf();

                newid_buf_map.insert(
                    {id,
                     sptree.compute_patch_owner(
                         shamsys::instance::get_compute_scheduler_ptr(),
                         pos,
                         xyz_field.get_obj_cnt())});
            }
        });
    }

    std::vector<std::unique_ptr<PatchData>> comm_pdat;
    std::vector<u64_2> comm_vec;

    sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
        if (!pdat.is_empty()) {

            sycl::buffer<u64> &newid = newid_buf_map.at(id);

            if (true) {

                // auto nid = newid.get_access<sycl::access::mode::read>();
                sycl::host_accessor nid{newid, sycl::read_only};
                std::unordered_map<u64, std::unique_ptr<PatchData>> send_map;

                const u32 cnt = pdat.get_obj_cnt();

                for (u32 i = cnt - 1; i < cnt; i--) {
                    if (id != nid[i]) {
                        // std::cout << id  << " " << i << " " << nid[i] << "\n";
                        std::unique_ptr<PatchData> &pdat_int = send_map[nid[i]];

                        if (!pdat_int) {
                            pdat_int = std::make_unique<PatchData>(sched.pdl);
                        }

                        pdat.extract_element(i, *pdat_int);
                    }

                } // std::cout << std::endl;

                for (auto &[receiver_pid, pdat_ptr] : send_map) {
                    // std::cout << "send " << id << " -> " << receiver_pid <<  " len : " <<
                    // pdat_ptr->pos_s.size()<<std::endl;

                    comm_vec.push_back(u64_2{
                        sched.patch_list.id_patch_to_global_idx[id],
                        sched.patch_list.id_patch_to_global_idx[receiver_pid]});
                    comm_pdat.push_back(std::move(pdat_ptr));
                }
            }
        }
    });

    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> part_xchg_map;
    for (u32 i = 0; i < comm_pdat.size(); i++) {

        shamlog_debug_ln(
            "PatchObjMover",
            comm_vec[i].x(),
            "->",
            comm_vec[i].y(),
            "data  size :",
            comm_pdat[i]->get_obj_cnt());

        PatchData &pdat = *comm_pdat[i];

        u32 ixyz = pdat.pdl.get_field_idx<f32_3>("xyz");

        /*
        for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
            print_vec(std::cout, pdat.fields_f32_3[ixyz].data()[i]);
            std::cout << std::endl;
        }
        */
    }

    patch_data_exchange_object(
        sched.pdl, sched.patch_list.global, comm_pdat, comm_vec, part_xchg_map);

    for (auto &[recv_id, vec_r] : part_xchg_map) {
        // std::cout << "patch " << recv_id << "\n";
        for (auto &[send_id, pdat] : vec_r) {
            // std::cout << "    " << send_id << " len : " << pdat->pos_s.size() << "\n";

            // TODO if crash here it means that this was implicit init => bad
            PatchData &pdat_recv = sched.patch_data.owned_data.get(recv_id);

            // std::cout << send_id << " -> " << recv_id << " recv data : " << std::endl;

            u32 ixyz = pdat->pdl.get_field_idx<f32_3>("xyz");

            /*
            for (u32 i = 0; i < pdat->get_obj_cnt(); i++) {
                print_vec(std::cout, pdat->fields_f32_3[ixyz].data()[i]);
                std::cout << std::endl;
            }
            */

            /*{
                std::cout << "recv : " << recv_id << " <- " << send_id << std::endl;

                std::cout << "cnt : " << pdat->pos_s.size() << std::endl;

                for(f32 a : pdat->U1_s){
                    std :: cout << a << " ";
                }std::cout << std::endl;

                for (u32 i = 0; i < pdat->pos_s.size(); i++) {

                    f32 val = pdat->U1_s[i*2 + 0];
                    if(val == 0){
                        std::cout << "----- fail id " << i  << " " << val << std::endl;
                        int a ;
                        std::cin >> a;
                    }
                }
            }*/

            //*
            pdat_recv.insert_elements(*pdat);
            //*/
        }
    }
}

template<>
[[deprecated("Legacy module")]]
inline void
reatribute_particles<f64_3>(PatchScheduler &sched, SerialPatchTree<f64_3> &sptree, bool periodic) {

    using namespace shamrock::patch;

    bool err_id_in_newid = false;
    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;
    sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
        if (!pdat.is_empty()) {

            u32 ixyz                         = sched.pdl.get_field_idx<f64_3>("xyz");
            PatchDataField<f64_3> &xyz_field = pdat.get_field<f64_3>(ixyz);

            if (xyz_field.get_nvar() != 1) {
                shambase::throw_unimplemented();
            }

            auto &pos = xyz_field.get_buf();

            newid_buf_map.insert(
                {id,
                 sptree.compute_patch_owner(
                     shamsys::instance::get_compute_scheduler_ptr(),
                     pos,
                     xyz_field.get_obj_cnt())});

            {
                // auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
                sycl::host_accessor nid{newid_buf_map.at(id), sycl::read_only};
                for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                    bool err        = nid[i] == u64_max;
                    err_id_in_newid = err_id_in_newid || (err);
                }
            }

            if (periodic && err_id_in_newid) {
                // auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
                sycl::host_accessor nid{newid_buf_map.at(id), sycl::read_only};
                for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                    bool err = nid[i] == u64_max;

                    if (periodic && err) {
                        logger::err_ln(
                            "Patch Object Mover", "id = ", i, "is out of bound with periodic mode");
                        throw shambase::make_except_with_loc<std::runtime_error>("error");
                    }
                }
            }
        }
    });

    shamlog_debug_ln("Patch Object Mover", "err_id_in_newid :", err_id_in_newid);

    bool synced_should_res_box = sched.should_resize_box(err_id_in_newid);

    if (periodic && synced_should_res_box) {

        throw shambase::make_except_with_loc<std::runtime_error>(
            "box cannot be resized in periodic mode");
    }

    if (synced_should_res_box) {
        sched.patch_data.sim_box.reset_box_size();

        auto [bmin_, bmax_] = sched.patch_data.sim_box.get_bounding_box<f64_3>();

        f64_3 bmin = bmin_;
        f64_3 bmax = bmax_;

        sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
            u32 ixyz                         = sched.pdl.get_field_idx<f64_3>("xyz");
            PatchDataField<f64_3> &xyz_field = pdat.get_field<f64_3>(ixyz);

            {

                auto &pos = xyz_field.get_buf();

                auto acc = pos.copy_to_stdvec();

                for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {

                    f64_3 r = acc[i];
                    bmin    = sycl::min(bmin, r);
                    bmax    = sycl::max(bmax, r);
                }
            }
        });

        sched.patch_data.sim_box.allreduce_set_bounding_box<f64_3>({bmin, bmax});
        sched.patch_data.sim_box.clean_box<f64>(1.2);

        shamlog_debug_ln(
            "Patch Object Mover",
            "resize box to  :",
            sched.patch_data.sim_box.get_bounding_box<f64_3>());

        sptree.detach_buf();
        sptree = SerialPatchTree<f64_3>(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<f64_3>());
        sptree.attach_buf();

        sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
            if (!pdat.is_empty()) {
                u32 ixyz                         = sched.pdl.get_field_idx<f64_3>("xyz");
                PatchDataField<f64_3> &xyz_field = pdat.get_field<f64_3>(ixyz);

                if (xyz_field.get_nvar() != 1) {
                    shambase::throw_unimplemented();
                }

                auto &pos = xyz_field.get_buf();

                newid_buf_map.insert(
                    {id,
                     sptree.compute_patch_owner(
                         shamsys::instance::get_compute_scheduler_ptr(),
                         pos,
                         xyz_field.get_obj_cnt())});
            }
        });
    }

    std::vector<std::unique_ptr<PatchData>> comm_pdat;
    std::vector<u64_2> comm_vec;

    sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
        if (!pdat.is_empty()) {

            sycl::buffer<u64> &newid = newid_buf_map.at(id);

            if (true) {

                // auto nid = newid.get_access<sycl::access::mode::read>();
                sycl::host_accessor nid{newid, sycl::read_only};
                std::unordered_map<u64, std::unique_ptr<PatchData>> send_map;

                const u32 cnt = pdat.get_obj_cnt();

                for (u32 i = cnt - 1; i < cnt; i--) {
                    if (id != nid[i]) {
                        // std::cout << id  << " " << i << " " << nid[i] << "\n";
                        std::unique_ptr<PatchData> &pdat_int = send_map[nid[i]];

                        if (!pdat_int) {
                            pdat_int = std::make_unique<PatchData>(sched.pdl);
                        }

                        pdat.extract_element(i, *pdat_int);
                    }

                } // std::cout << std::endl;

                for (auto &[receiver_pid, pdat_ptr] : send_map) {
                    // std::cout << "send " << id << " -> " << receiver_pid <<  " len : " <<
                    // pdat_ptr->pos_s.size()<<std::endl;

                    comm_vec.push_back(u64_2{
                        sched.patch_list.id_patch_to_global_idx[id],
                        sched.patch_list.id_patch_to_global_idx[receiver_pid]});
                    comm_pdat.push_back(std::move(pdat_ptr));
                }
            }
        }
    });

    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> part_xchg_map;
    for (u32 i = 0; i < comm_pdat.size(); i++) {

        shamlog_debug_ln(
            "PatchObjMover",
            comm_vec[i].x(),
            "->",
            comm_vec[i].y(),
            "data  size :",
            comm_pdat[i]->get_obj_cnt());

        PatchData &pdat = *comm_pdat[i];

        u32 ixyz = pdat.pdl.get_field_idx<f64_3>("xyz");

        /*
        for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
            print_vec(std::cout, pdat.fields_f64_3[ixyz].data()[i]);
            std::cout << std::endl;
        }
        */
    }

    patch_data_exchange_object(
        sched.pdl, sched.patch_list.global, comm_pdat, comm_vec, part_xchg_map);

    for (auto &[recv_id, vec_r] : part_xchg_map) {
        // std::cout << "patch " << recv_id << "\n";
        for (auto &[send_id, pdat] : vec_r) {
            // std::cout << "    " << send_id << " len : " << pdat->pos_s.size() << "\n";

            // TODO if crash here it means that this was implicit init => bad
            PatchData &pdat_recv = sched.patch_data.owned_data.get(recv_id);

            // std::cout << send_id << " -> " << recv_id << " recv data : " << std::endl;

            u32 ixyz = pdat->pdl.get_field_idx<f64_3>("xyz");

            /*
            for (u32 i = 0; i < pdat->get_obj_cnt(); i++) {
                print_vec(std::cout, pdat->fields_f64_3[ixyz].data()[i]);
                std::cout << std::endl;
            }
            */

            /*{
                std::cout << "recv : " << recv_id << " <- " << send_id << std::endl;

                std::cout << "cnt : " << pdat->pos_s.size() << std::endl;

                for(f64 a : pdat->U1_s){
                    std :: cout << a << " ";
                }std::cout << std::endl;

                for (u32 i = 0; i < pdat->pos_s.size(); i++) {

                    f64 val = pdat->U1_s[i*2 + 0];
                    if(val == 0){
                        std::cout << "----- fail id " << i  << " " << val << std::endl;
                        int a ;
                        std::cin >> a;
                    }
                }
            }*/

            //*
            pdat_recv.insert_elements(*pdat);
            //*/
        }
    }
}
