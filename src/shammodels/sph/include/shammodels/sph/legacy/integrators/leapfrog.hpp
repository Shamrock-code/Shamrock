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
 * @file leapfrog.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
#include "shamrock/legacy/io/dump.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/comm/patch_object_mover.hpp"
#include "shamrock/legacy/patch/interfaces/interface_handler.hpp"
#include "shamrock/legacy/patch/interfaces/interface_selector.hpp"
#include "shamrock/legacy/patch/utility/compute_field.hpp"
#include "shamrock/legacy/patch/utility/global_var.hpp"
#include "shamrock/legacy/patch/utility/merged_patch.hpp"
#include "shamrock/legacy/utils/syclreduction.hpp"
#include "shamsys/legacy/log.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shammodels/sph/legacy/algs/smoothing_length.hpp"
#include "shammodels/sph/legacy/sphpatch.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shamrock/sph/sphpart.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtree/RadixTree.hpp"
#include <unordered_map>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

//%Impl status : Good

namespace integrators {

    namespace sph {

        template<class flt, class Kernel, class u_morton>
        class LeapfrogGeneral {
            public:
            using vec3 = sycl::vec<flt, 3>;

            static_assert(
                std::is_same<flt, f16>::value || std::is_same<flt, f32>::value
                    || std::is_same<flt, f64>::value,
                "Leapfrog : floating point type should be one of (f16,f32,f64)");

            inline static void sycl_move_parts(
                sycl::queue &queue,
                u32 npart,
                flt dt,
                const std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                const std::unique_ptr<sycl::buffer<vec3>> &buf_vxyz) {

                sycl::range<1> range_npart{npart};

                auto ker_predict_step = [&](sycl::handler &cgh) {
                    auto acc_xyz
                        = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);
                    auto acc_vxyz
                        = buf_vxyz->template get_access<sycl::access::mode::read_write>(cgh);

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 gid = (u32) item.get_id();

                        vec3 &vxyz = acc_vxyz[item];

                        acc_xyz[item] = acc_xyz[item] + dt * vxyz;
                    });
                };

                queue.submit(ker_predict_step);
            }

            inline static void sycl_position_modulo(
                sycl::queue &queue,
                u32 npart,
                const std::unique_ptr<sycl::buffer<vec3>> &buf_xyz,
                std::tuple<vec3, vec3> box) {

                sycl::range<1> range_npart{npart};

                auto ker_predict_step = [&](sycl::handler &cgh) {
                    auto xyz = buf_xyz->template get_access<sycl::access::mode::read_write>(cgh);

                    vec3 box_min = std::get<0>(box);
                    vec3 box_max = std::get<1>(box);
                    vec3 delt    = box_max - box_min;

                    // Executing kernel
                    cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                        u32 gid = (u32) item.get_id();

                        vec3 r = xyz[gid] - box_min;

                        r = sycl::fmod(r, delt);
                        r += delt;
                        r = sycl::fmod(r, delt);
                        r += box_min;

                        xyz[gid] = r;
                    });
                };

                queue.submit(ker_predict_step);
            }

            // mandatory variables
            PatchScheduler &sched;
            bool periodic_mode;
            flt htol_up_tol;
            flt htol_up_iter;

            flt sph_gpart_mass;

            LeapfrogGeneral(
                PatchScheduler &sched,
                bool periodic_mode,
                flt htol_up_tol,
                flt htol_up_iter,
                flt sph_gpart_mass)
                : sched(sched), periodic_mode(periodic_mode), htol_up_tol(htol_up_tol),
                  htol_up_iter(htol_up_iter), sph_gpart_mass(sph_gpart_mass) {}

            template<
                class LambdaCFL,
                class LambdaUpdateTime,
                class LambdaSwapDer,
                class LambdaPostSync,
                class LambdaForce,
                class LambdaCorrector>
            inline flt step(
                flt old_time,
                bool do_force,
                bool do_corrector,
                LambdaCFL &&lambda_cfl,
                LambdaUpdateTime &&lambda_update_time,
                LambdaSwapDer &&lambda_swap_der,
                LambdaPostSync &&lambda_post_sync,
                LambdaForce &&lambda_compute_forces,
                LambdaCorrector &&lambda_correct) {

                using namespace shamrock::patch;

                const flt loc_htol_up_tol  = htol_up_tol;
                const flt loc_htol_up_iter = htol_up_iter;

                const u32 ixyz      = sched.pdl.get_field_idx<vec3>("xyz");
                const u32 ivxyz     = sched.pdl.get_field_idx<vec3>("vxyz");
                const u32 iaxyz     = sched.pdl.get_field_idx<vec3>("axyz");
                const u32 iaxyz_old = sched.pdl.get_field_idx<vec3>("axyz_old");

                const u32 ihpart = sched.pdl.get_field_idx<flt>("hpart");

                logger::info_ln(
                    "SPHLeapfrog",
                    "step t=",
                    old_time,
                    "do_force =",
                    do_force,
                    "do_corrector =",
                    do_corrector);

                // Init serial patch tree
                SerialPatchTree<vec3> sptree(
                    sched.patch_tree, sched.get_sim_box().template get_patch_transform<vec3>());
                sptree.attach_buf();

                // compute cfl
                auto get_cfl = [&]() -> flt {
                    GlobalVariable<GlobalVariableType::min, flt> cfl_glb_var;
                    cfl_glb_var.compute_var_patch(sched, lambda_cfl);
                    cfl_glb_var.reduce_val();
                    return cfl_glb_var.get_val();
                };

                flt cfl_val = get_cfl();

                // compute dt step

                flt dt_cur = cfl_val;

                logger::info_ln("SPHLeapfrog", "current dt  :", dt_cur);

                // advance time
                flt step_time = old_time;
                step_time += dt_cur;

                // leapfrog predictor
                sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                    shamlog_debug_ln("SPHLeapfrog", "patch : n°", id_patch, "->", "predictor");

                    lambda_update_time(
                        shamsys::instance::get_compute_queue(),
                        pdat,
                        sycl::range<1>{pdat.get_obj_cnt()},
                        dt_cur / 2);

                    sycl_move_parts(
                        shamsys::instance::get_compute_queue(),
                        pdat.get_obj_cnt(),
                        dt_cur,
                        pdat.get_field<vec3>(ixyz).get_buf(),
                        pdat.get_field<vec3>(ivxyz).get_buf());

                    lambda_update_time(
                        shamsys::instance::get_compute_queue(),
                        pdat,
                        sycl::range<1>{pdat.get_obj_cnt()},
                        dt_cur / 2);

                    shamlog_debug_ln("SPHLeapfrog", "patch : n°", id_patch, "->", "dt fields swap");

                    lambda_swap_der(
                        shamsys::instance::get_compute_queue(),
                        pdat,
                        sycl::range<1>{pdat.get_obj_cnt()});

                    if (periodic_mode) { // TODO generalise position modulo in the scheduler
                        sycl_position_modulo(
                            shamsys::instance::get_compute_queue(),
                            pdat.get_obj_cnt(),
                            pdat.get_field<vec3>(ixyz).get_buf(),
                            sched.get_box_volume<vec3>());
                    }
                });

                // move particles between patches
                shamlog_debug_ln("SPHLeapfrog", "particle reatribution");
                reatribute_particles(sched, sptree, periodic_mode);

                shamlog_debug_ln("SPHLeapfrog", "compute hmax of each patches");

                // compute hmax
                legacy::PatchField<flt> h_field;
                // sched.compute_patch_field(
                //     h_field, get_mpi_type<flt>(), [loc_htol_up_tol](sycl::queue &queue, Patch &p,
                //     PatchDataBuffer &pdat_buf) {
                //         return patchdata::sph::get_h_max<flt>(pdat_buf.pdl, queue, pdat_buf) *
                //         loc_htol_up_tol * Kernel::Rkern;
                //     });

                sched.compute_patch_field(
                    h_field,
                    get_mpi_type<flt>(),
                    [loc_htol_up_tol](sycl::queue &queue, Patch &p, PatchData &pdat) {
                        return patchdata::sph::get_h_max<flt>(pdat.pdl, queue, pdat)
                               * loc_htol_up_tol * Kernel::Rkern;
                    });

                shamlog_debug_ln("SPHLeapfrog", "compute interface list");
                // make interfaces
                LegacyInterfacehandler<vec3, flt> interface_hndl;
                interface_hndl.template compute_interface_list<InterfaceSelector_SPH<vec3, flt>>(
                    sched, sptree, h_field, periodic_mode);

                shamlog_debug_ln("SPHLeapfrog", "communicate interfaces");
                interface_hndl.comm_interfaces(sched, periodic_mode);

                shamlog_debug_ln("SPHLeapfrog", "merging interfaces with data");
                // merging strategy

                // old
                // std::unordered_map<u64, MergedPatchDataBuffer<vec3>> merge_pdat_buf;
                // make_merge_patches(sched, interface_hndl, merge_pdat_buf);

                std::unordered_map<u64, MergedPatchData<flt>> merge_pdat
                    = MergedPatchData<flt>::merge_patches(sched, interface_hndl);

                shamsys::instance::get_compute_queue().wait();

                constexpr u32 reduc_level = 5;

                // make trees
                std::unordered_map<u64, std::unique_ptr<RadixTree<u_morton, vec3>>> radix_trees;
                std::unordered_map<u64, std::unique_ptr<RadixTreeField<flt>>> cell_int_rads;

                sched.for_each_patch([&](u64 id_patch, Patch /*cur_p*/) {
                    shamlog_debug_ln(
                        "SPHLeapfrog", "patch : n°", id_patch, "->", "making Radix Tree");

                    if (merge_pdat.at(id_patch).or_element_cnt == 0)
                        shamlog_debug_ln(
                            "SPHLeapfrog",
                            "patch : n°",
                            id_patch,
                            "->",
                            "is empty skipping tree build");

                    // PatchDataBuffer &mpdat_buf = *merge_pdat_buf.at(id_patch).data;
                    PatchData &mpdat = merge_pdat.at(id_patch).data;

                    auto &buf_xyz = mpdat.get_field<vec3>(ixyz).get_buf();

                    std::tuple<vec3, vec3> &box = merge_pdat.at(id_patch).box;

                    // radix tree computation
                    radix_trees[id_patch] = std::make_unique<RadixTree<u_morton, vec3>>(
                        shamsys::instance::get_compute_queue(),
                        box,
                        buf_xyz,
                        mpdat.get_obj_cnt(),
                        reduc_level);
                });

                sched.for_each_patch([&](u64 id_patch, Patch /*cur_p*/) {
                    shamlog_debug_ln(
                        "SPHLeapfrog",
                        "patch : n°",
                        id_patch,
                        "->",
                        "compute radix tree cell volumes");
                    if (merge_pdat.at(id_patch).or_element_cnt == 0)
                        shamlog_debug_ln(
                            "SPHLeapfrog",
                            "patch : n°",
                            id_patch,
                            "->",
                            "is empty skipping tree volumes step");

                    radix_trees[id_patch]->compute_cell_ibounding_box(
                        shamsys::instance::get_compute_queue());
                    radix_trees[id_patch]->convert_bounding_box(
                        shamsys::instance::get_compute_queue());
                });

                sched.for_each_patch([&](u64 id_patch, Patch /*cur_p*/) {
                    shamlog_debug_ln(
                        "SPHLeapfrog",
                        "patch : n°",
                        id_patch,
                        "->",
                        "compute Radix Tree interaction boxes");
                    if (merge_pdat.at(id_patch).or_element_cnt == 0)
                        shamlog_debug_ln(
                            "SPHLeapfrog",
                            "patch : n°",
                            id_patch,
                            "->",
                            "is empty skipping interaction box compute");

                    PatchData &mpdat = merge_pdat.at(id_patch).data;

                    auto &buf_h = mpdat.get_field<flt>(ihpart).get_buf();

                    cell_int_rads[id_patch] = std::make_unique<RadixTreeField<flt>>(
                        radix_trees[id_patch]->compute_int_boxes(
                            shamsys::instance::get_compute_queue(), buf_h, htol_up_tol));
                });
                shamsys::instance::get_compute_queue().wait();

                // create compute field for new h and omega
                shamlog_debug_ln("SPHLeapfrog", "init compute fields : hnew, omega");

                PatchComputeField<flt> hnew_field;
                PatchComputeField<flt> omega_field;

                hnew_field.generate(sched);
                omega_field.generate(sched);

                // iterate smoothing length
                sched.for_each_patch([&](u64 id_patch, Patch /*cur_p*/) {
                    shamlog_debug_ln(
                        "SPHLeapfrog", "patch : n°", id_patch, "->", "Init h iteration");
                    if (merge_pdat.at(id_patch).or_element_cnt == 0)
                        shamlog_debug_ln(
                            "SPHLeapfrog",
                            "patch : n°",
                            id_patch,
                            "->",
                            "is empty skipping h iteration");

                    PatchData &pdat_merge = merge_pdat.at(id_patch).data;

                    auto &hnew  = hnew_field.get_buf(id_patch);
                    auto &omega = omega_field.get_buf(id_patch);
                    sycl::buffer<flt> eps_h
                        = sycl::buffer<flt>(merge_pdat.at(id_patch).or_element_cnt);

                    sycl::range range_npart{merge_pdat.at(id_patch).or_element_cnt};

                    shamlog_debug_ln(
                        "SPHLeapfrog",
                        "merging -> original size :",
                        merge_pdat.at(id_patch).or_element_cnt,
                        "| merged :",
                        pdat_merge.get_obj_cnt());

                    models::sph::algs::SmoothinglengthCompute<flt, u32, Kernel> h_iterator(
                        sched.pdl, htol_up_tol, htol_up_iter);

                    h_iterator.iterate_smoothing_length(
                        shamsys::instance::get_compute_queue(),
                        merge_pdat.at(id_patch).or_element_cnt,
                        sph_gpart_mass,
                        *radix_trees[id_patch],
                        *cell_int_rads[id_patch],
                        pdat_merge,
                        *hnew,
                        *omega,
                        eps_h);

                    // write back h test
                    //*
                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        auto h_new = hnew->template get_access<sycl::access::mode::read>(cgh);

                        auto acc_hpart = pdat_merge.get_field<flt>(ihpart)
                                             .get_buf()
                                             ->template get_access<sycl::access::mode::write>(cgh);

                        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                            acc_hpart[item] = h_new[item];
                        });
                    });
                    //*/
                });

                shamlog_debug_ln("SPHLeapfrog", "exchange interface hnew");
                PatchComputeFieldInterfaces<flt> hnew_field_interfaces
                    = interface_hndl.template comm_interfaces_field<flt>(
                        sched, hnew_field, periodic_mode);
                shamlog_debug_ln("SPHLeapfrog", "exchange interface omega");
                PatchComputeFieldInterfaces<flt> omega_field_interfaces
                    = interface_hndl.template comm_interfaces_field<flt>(
                        sched, omega_field, periodic_mode);

                // merge compute fields
                // std::unordered_map<u64, MergedPatchCompFieldBuffer<flt>> hnew_field_merged;
                // make_merge_patches_comp_field<flt>(sched, interface_hndl, hnew_field,
                // hnew_field_interfaces, hnew_field_merged); std::unordered_map<u64,
                // MergedPatchCompFieldBuffer<flt>> omega_field_merged;
                // make_merge_patches_comp_field<flt>(sched, interface_hndl, omega_field,
                // omega_field_interfaces, omega_field_merged);

                std::unordered_map<u64, MergedPatchCompField<flt, flt>> hnew_field_merged
                    = MergedPatchCompField<flt, flt>::merge_patches_cfield(
                        sched, interface_hndl, hnew_field, hnew_field_interfaces);

                std::unordered_map<u64, MergedPatchCompField<flt, flt>> omega_field_merged
                    = MergedPatchCompField<flt, flt>::merge_patches_cfield(
                        sched, interface_hndl, omega_field, omega_field_interfaces);

                // TODO add looping on corrector step

                lambda_post_sync(sched, merge_pdat, hnew_field_merged, omega_field_merged);

                // compute force
                if (do_force) {
                    lambda_compute_forces(
                        sched,
                        radix_trees,
                        cell_int_rads,
                        merge_pdat,
                        hnew_field_merged,
                        omega_field_merged,
                        htol_up_tol);
                }

                // leapfrog corrector
                sched.for_each_patch([&](u64 id_patch, Patch /*cur_p*/) {
                    if (merge_pdat.at(id_patch).or_element_cnt == 0) {
                        std::cout << " empty => skipping" << std::endl;
                        return;
                    }

                    PatchData &pdat_merge = merge_pdat.at(id_patch).data;

                    if (do_corrector) {

                        shamlog_debug_ln("SPHLeapfrog", "leapfrog corrector");

                        lambda_correct(
                            shamsys::instance::get_compute_queue(),
                            pdat_merge,
                            sycl::range<1>{merge_pdat.at(id_patch).or_element_cnt},
                            dt_cur / 2);
                    }
                });

                // write_back_merge_patches(sched, interface_hndl, merge_pdat);
                write_back_merge_patches(sched, merge_pdat);

                return step_time;
            }
        };

    } // namespace sph

} // namespace integrators
