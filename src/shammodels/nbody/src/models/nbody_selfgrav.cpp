// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file nbody_selfgrav.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/nbody/models/nbody_selfgrav.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shamphys/fmm.hpp"
#include "shamphys/fmm/GreenFuncGravCartesian.hpp"
#include "shamphys/fmm/offset_multipole.hpp"
#include "shamrock/legacy/patch/comm/patch_object_mover.hpp"
#include "shamrock/legacy/patch/interfaces/interface_handler.hpp"
#include "shamrock/legacy/patch/utility/full_tree_field.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamtree/RadixTree.hpp"

const std::string console_tag = "[NBodySelfGrav] ";

constexpr u32 fmm_order = 4;

template<class flt>
void models::nbody::Nbody_SelfGrav<flt>::check_valid() {

    if (cfl_force < 0) {
        throw ShamAPIException(console_tag + "cfl force not set");
    }

    if (gpart_mass < 0) {
        throw ShamAPIException(console_tag + "particle mass not set");
    }
}

template<class flt>
void models::nbody::Nbody_SelfGrav<flt>::init() {}

template<class flt, class vec3>
void sycl_move_parts(
    sham::DeviceQueue &queue,
    u32 npart,
    flt dt,
    sham::DeviceBuffer<vec3> &buf_xyz,
    sham::DeviceBuffer<vec3> &buf_vxyz) {

    using namespace shamrock::patch;

    sycl::range<1> range_npart{npart};

    sham::EventList depends_list;
    auto acc_xyz  = buf_xyz.get_write_access(depends_list);
    auto acc_vxyz = buf_vxyz.get_read_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        // Executing kernel
        cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id();

            vec3 vxyz = acc_vxyz[item];

            acc_xyz[item] = acc_xyz[item] + dt * vxyz;
        });
    });

    buf_xyz.complete_event_state(e);
    buf_vxyz.complete_event_state(e);
}

template<class vec3>
void sycl_position_modulo(
    sham::DeviceQueue &queue,
    u32 npart,
    sham::DeviceBuffer<vec3> &buf_xyz,
    std::tuple<vec3, vec3> box) {

    sycl::range<1> range_npart{npart};

    sham::EventList depends_list;
    auto xyz = buf_xyz.get_write_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
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
    });

    buf_xyz.complete_event_state(e);
}

template<class flt>
class FMMInteract_cd {

    using vec = sycl::vec<flt, 3>;

    flt opening_crit_sq;

    public:
    explicit FMMInteract_cd(flt open_crit) : opening_crit_sq(open_crit * open_crit) {};

    static bool interact_cd_cell_cell(
        const FMMInteract_cd &cd, vec b1_min, vec b1_max, vec b2_min, vec b2_max) {
        vec s1 = (b1_max + b1_min) / 2;
        vec s2 = (b2_max + b2_min) / 2;

        vec r_fmm = s2 - s1;

        vec d1 = b1_max - b1_min;
        vec d2 = b2_max - b2_min;

        flt l1 = sycl::max(sycl::max(d1.x(), d1.y()), d1.z());
        flt l2 = sycl::max(sycl::max(d2.x(), d2.y()), d2.z());

        flt opening_angle_sq = (l1 + l2) * (l1 + l2) / sycl::dot(r_fmm, r_fmm);

        return opening_angle_sq > cd.opening_crit_sq;
    }

    static bool interact_cd_cell_patch(
        const FMMInteract_cd &cd,
        vec b1_min,
        vec b1_max,
        vec b2_min,
        vec b2_max,
        flt b1_min_slength,
        flt b1_max_slength,
        flt b2_min_slength,
        flt b2_max_slength) {

        // return true;
        // return interact_cd_cell_cell(cd, b1_min, b1_max, b2_min, b2_max);

        vec c1 = (b1_max + b1_min) / 2;
        vec s1 = (b1_max - b1_min);
        flt L1 = sycl::max(sycl::max(s1.x(), s1.y()), s1.z());

        flt dist_to_surf = sycl::sqrt(BBAA::get_sq_distance_to_BBAAsurface(c1, b2_min, b2_max));

        flt opening_angle_sq = (L1 + b2_max_slength) / (dist_to_surf /*+ b2_min_slength/2*/);
        opening_angle_sq *= opening_angle_sq;

        return opening_angle_sq > cd.opening_crit_sq;
    }

    static bool interact_cd_cell_patch_outdomain(
        const FMMInteract_cd &cd,
        vec b1_min,
        vec b1_max,
        vec b2_min,
        vec b2_max,
        flt b1_min_slength,
        flt b1_max_slength,
        flt b2_min_slength,
        flt b2_max_slength) {
        return false;
    }
};

template<class Tree, class vec, class flt>
void compute_multipoles(
    Tree &rtree,
    sham::DeviceBuffer<vec> &pos_part,
    sycl::buffer<flt> &grav_multipoles,
    flt gpart_mass) {

    using namespace shammath;

    shamlog_debug_sycl_ln(
        "RTreeFMM",
        "computing leaf moments (",
        rtree.tree_reduced_morton_codes.tree_leaf_count,
        ")");

    sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();

    sham::EventList depends_list;
    auto xyz = pos_part.get_read_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        u32 offset_leaf = rtree.tree_struct.internal_cell_count;

        auto cell_particle_ids = sycl::accessor{
            *rtree.tree_reduced_morton_codes.buf_reduc_index_map, cgh, sycl::read_only};
        auto particle_index_map
            = sycl::accessor{*rtree.tree_morton_codes.buf_particle_index_map, cgh, sycl::read_only};
        auto cell_max
            = sycl::accessor{*rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};
        auto cell_min
            = sycl::accessor{*rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
        auto multipoles = sycl::accessor{grav_multipoles, cgh, sycl::write_only, sycl::no_init};

        sycl::range<1> range_leaf_cell{rtree.tree_reduced_morton_codes.tree_leaf_count};

        const flt m = gpart_mass;

        cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id(0);

            u32 min_ids = cell_particle_ids[gid];
            u32 max_ids = cell_particle_ids[gid + 1];

            vec cell_pmax = cell_max[offset_leaf + gid];
            vec cell_pmin = cell_min[offset_leaf + gid];

            vec s_b = (cell_pmax + cell_pmin) / 2;

            auto B_n = SymTensorCollection<flt, 0, fmm_order>::zeros();

            for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                u32 idx_j = particle_index_map[id_s];
                vec bj    = xyz[idx_j] - s_b;

                auto tB_n = SymTensorCollection<flt, 0, fmm_order>::from_vec(bj);

                const flt m_j = m;

                tB_n *= m_j;
                B_n += tB_n;
            }

            B_n.store(
                multipoles,
                (gid + offset_leaf) * SymTensorCollection<flt, 0, fmm_order>::num_component);
        });
    });

    pos_part.complete_event_state(e);

    auto buf_is_computed = std::make_unique<sycl::buffer<u8>>(
        (rtree.tree_struct.internal_cell_count + rtree.tree_reduced_morton_codes.tree_leaf_count));

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        auto is_computed = sycl::accessor{*buf_is_computed, cgh, sycl::write_only, sycl::no_init};
        sycl::range<1> range_internal_count{
            rtree.tree_struct.internal_cell_count
            + rtree.tree_reduced_morton_codes.tree_leaf_count};

        u32 int_cnt = rtree.tree_struct.internal_cell_count;

        cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {
            is_computed[item] = item.get_linear_id() >= int_cnt;
        });
    });

    for (u32 iter = 0; iter < rtree.tree_depth; iter++) {

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            u32 leaf_offset = rtree.tree_struct.internal_cell_count;

            auto cell_max = sycl::accessor{
                *rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};
            auto cell_min = sycl::accessor{
                *rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
            auto multipoles  = sycl::accessor{grav_multipoles, cgh, sycl::read_write};
            auto is_computed = sycl::accessor{*buf_is_computed, cgh, sycl::read_write};

            sycl::range<1> range_internal_count{rtree.tree_struct.internal_cell_count};

            auto rchild_id = sycl::accessor{*rtree.tree_struct.buf_rchild_id, cgh, sycl::read_only};
            auto lchild_id = sycl::accessor{*rtree.tree_struct.buf_lchild_id, cgh, sycl::read_only};
            auto rchild_flag
                = sycl::accessor{*rtree.tree_struct.buf_rchild_flag, cgh, sycl::read_only};
            auto lchild_flag
                = sycl::accessor{*rtree.tree_struct.buf_lchild_flag, cgh, sycl::read_only};

            cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {
                u32 cid = item.get_linear_id();

                u32 lid = lchild_id[cid] + leaf_offset * lchild_flag[cid];
                u32 rid = rchild_id[cid] + leaf_offset * rchild_flag[cid];

                bool should_compute = (!is_computed[cid]) && (is_computed[lid] && is_computed[rid]);

                if (should_compute) {

                    vec cell_pmax = cell_max[cid];
                    vec cell_pmin = cell_min[cid];

                    vec sbp = (cell_pmax + cell_pmin) / 2;

                    auto B_n = SymTensorCollection<flt, 0, fmm_order>::zeros();

                    auto add_multipole_offset = [&](u32 s_cid) {
                        vec s_cell_pmax = cell_max[s_cid];
                        vec s_cell_pmin = cell_min[s_cid];

                        vec sb = (s_cell_pmax + s_cell_pmin) / 2;

                        auto d = sb - sbp;

                        auto B_ns = SymTensorCollection<flt, 0, fmm_order>::load(
                            multipoles,
                            s_cid * SymTensorCollection<flt, 0, fmm_order>::num_component);

                        auto B_ns_offseted = shamphys::offset_multipole(B_ns, d);

                        B_n += B_ns_offseted;
                    };

                    add_multipole_offset(lid);
                    add_multipole_offset(rid);

                    is_computed[cid] = true;
                    B_n.store(
                        multipoles, cid * SymTensorCollection<flt, 0, fmm_order>::num_component);
                }
            });
        });
    }
}

template<class T, class flt>
inline void field_advance_time(
    sham::DeviceQueue &queue,
    sham::DeviceBuffer<T> &buf_val,
    sham::DeviceBuffer<T> &buf_der,
    sycl::range<1> elem_range,
    flt dt) {

    sham::EventList depends_list;

    auto acc_u  = buf_val.get_write_access(depends_list);
    auto acc_du = buf_der.get_read_access(depends_list);

    auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
        // Executing kernel
        cgh.parallel_for(elem_range, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id();

            T du = acc_du[item];

            acc_u[item] = acc_u[item] + (dt) * (du);
        });
    });

    buf_val.complete_event_state(e);
    buf_der.complete_event_state(e);
}

template<class flt>
f64 models::nbody::Nbody_SelfGrav<flt>::evolve(
    PatchScheduler &sched, f64 old_time, f64 target_time) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shammath;

    check_valid();

    logger::info_ln("NBodySelfGrav", "evolve t=", old_time);

    // Stepper stepper(sched,periodic_bc,htol_up_tol,htol_up_iter,gpart_mass);

    const u32 ixyz      = sched.pdl().get_field_idx<vec3>("xyz");
    const u32 ivxyz     = sched.pdl().get_field_idx<vec3>("vxyz");
    const u32 iaxyz     = sched.pdl().get_field_idx<vec3>("axyz");
    const u32 iaxyz_old = sched.pdl().get_field_idx<vec3>("axyz_old");

    // const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");

    // PatchComputeField<f32> pressure_field;

    auto lambda_update_time
        = [&](sham::DeviceQueue &queue, PatchDataLayer &pdat, sycl::range<1> range_npart, flt hdt) {
              sham::DeviceBuffer<vec3> &vxyz = pdat.get_field<vec3>(ivxyz).get_buf();
              sham::DeviceBuffer<vec3> &axyz = pdat.get_field<vec3>(iaxyz).get_buf();

              field_advance_time(queue, vxyz, axyz, range_npart, hdt);
          };

    auto lambda_swap_der
        = [&](sham::DeviceQueue &queue, PatchDataLayer &pdat, sycl::range<1> range_npart) {
              sham::EventList depends_list;

              auto acc_axyz = pdat.get_field<vec3>(iaxyz).get_buf().get_write_access(depends_list);
              auto acc_axyz_old
                  = pdat.get_field<vec3>(iaxyz_old).get_buf().get_write_access(depends_list);

              auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
                  // Executing kernel
                  cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                      vec3 axyz     = acc_axyz[item];
                      vec3 axyz_old = acc_axyz_old[item];

                      acc_axyz[item]     = vec3{0, 0, 0};
                      acc_axyz_old[item] = axyz;
                  });
              });

              pdat.get_field<vec3>(iaxyz).get_buf().complete_event_state(e);
              pdat.get_field<vec3>(iaxyz_old).get_buf().complete_event_state(e);
          };

    auto lambda_correct =
        [&](sham::DeviceQueue &queue, PatchDataLayer &buf, sycl::range<1> range_npart, flt hdt) {
            sham::DeviceBuffer<vec3> &vxyz     = buf.get_field<vec3>(ivxyz).get_buf();
            sham::DeviceBuffer<vec3> &axyz     = buf.get_field<vec3>(iaxyz).get_buf();
            sham::DeviceBuffer<vec3> &axyz_old = buf.get_field<vec3>(iaxyz_old).get_buf();

            sham::EventList depends_list;

            auto acc_vxyz     = vxyz.get_write_access(depends_list);
            auto acc_axyz     = axyz.get_write_access(depends_list);
            auto acc_axyz_old = axyz_old.get_write_access(depends_list);

            auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
                // Executing kernel
                cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                    // u32 gid = (u32)item.get_id();
                    //
                    // vec3 &vxyz     = acc_vxyz[item];
                    // vec3 &axyz     = acc_axyz[item];
                    // vec3 &axyz_old = acc_axyz_old[item];

                    // v^* = v^{n + 1/2} + dt/2 a^n
                    acc_vxyz[item] = acc_vxyz[item] + (hdt) * (acc_axyz[item] - acc_axyz_old[item]);
                });
            });

            vxyz.complete_event_state(e);
            axyz.complete_event_state(e);
            axyz_old.complete_event_state(e);
        };

    auto leapfrog_lambda = [&](flt old_time, bool do_force, bool do_corrector) -> flt {
        const u32 ixyz      = sched.pdl().get_field_idx<vec3>("xyz");
        const u32 ivxyz     = sched.pdl().get_field_idx<vec3>("vxyz");
        const u32 iaxyz     = sched.pdl().get_field_idx<vec3>("axyz");
        const u32 iaxyz_old = sched.pdl().get_field_idx<vec3>("axyz_old");

        logger::info_ln(
            "NBodyleapfrog",
            "step t=",
            old_time,
            "do_force =",
            do_force,
            "do_corrector =",
            do_corrector);

        // Init serial patch tree
        SerialPatchTree<vec3> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<vec3>());
        sptree.dump_dat();
        sptree.attach_buf();

        // compute cfl
        flt cfl_val = 1e-2;

        // compute dt step

        flt dt_cur = cfl_val;

        logger::info_ln("SPHLeapfrog", "current dt  :", dt_cur);

        // advance time
        flt step_time = old_time;
        step_time += dt_cur;

        // leapfrog predictor
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            shamlog_debug_ln("SPHLeapfrog", "patch : n°", id_patch, "->", "predictor");

            lambda_update_time(
                shamsys::instance::get_compute_scheduler().get_queue(),
                pdat,
                sycl::range<1>{pdat.get_obj_cnt()},
                dt_cur / 2);

            sycl_move_parts(
                shamsys::instance::get_compute_scheduler().get_queue(),
                pdat.get_obj_cnt(),
                dt_cur,
                pdat.get_field<vec3>(ixyz).get_buf(),
                pdat.get_field<vec3>(ivxyz).get_buf());

            lambda_update_time(
                shamsys::instance::get_compute_scheduler().get_queue(),
                pdat,
                sycl::range<1>{pdat.get_obj_cnt()},
                dt_cur / 2);

            shamlog_debug_ln("SPHLeapfrog", "patch : n°", id_patch, "->", "dt fields swap");

            lambda_swap_der(
                shamsys::instance::get_compute_scheduler().get_queue(),
                pdat,
                sycl::range<1>{pdat.get_obj_cnt()});

            if (periodic_bc) { // TODO generalise position modulo in the scheduler
                sycl_position_modulo(
                    shamsys::instance::get_compute_scheduler().get_queue(),
                    pdat.get_obj_cnt(),
                    pdat.get_field<vec3>(ixyz).get_buf(),
                    sched.get_box_volume<vec3>());
            }
        });

        // move particles between patches
        shamlog_debug_ln("SPHLeapfrog", "particle reatribution");
        reatribute_particles(sched, sptree, periodic_bc);

        constexpr u32 reduc_level = 2;

        using RadTree = RadixTree<u_morton, vec3>;

        // make trees
        std::unordered_map<u64, std::unique_ptr<RadTree>> radix_trees;

        sched.for_each_patch_data([&](u64 id_patch, Patch &cur_p, PatchDataLayer &pdat) {
            shamlog_debug_ln(
                "SPHLeapfrog",
                "patch : n°",
                id_patch,
                "->",
                "making Radix Tree ( N=",
                pdat.get_obj_cnt(),
                ")");

            if (pdat.is_empty()) {
                shamlog_debug_ln(
                    "SPHLeapfrog", "patch : n°", id_patch, "->", "is empty skipping tree build");
            } else {

                auto &buf_xyz = pdat.get_field<vec3>(ixyz).get_buf();

                std::tuple<vec3, vec3> box = sched.patch_data.sim_box.get_box<flt>(cur_p);

                // radix tree computation
                radix_trees[id_patch] = std::make_unique<RadTree>(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    box,
                    buf_xyz,
                    pdat.get_obj_cnt(),
                    reduc_level);
            }
        });

        sched.for_each_patch_data([&](u64 id_patch, Patch & /*cur_p*/, PatchDataLayer &pdat) {
            shamlog_debug_ln(
                "SPHLeapfrog", "patch : n°", id_patch, "->", "compute radix tree cell volumes");
            if (pdat.is_empty()) {
                shamlog_debug_ln(
                    "SPHLeapfrog", "patch : n°", id_patch, "->", "is empty skipping tree build");
            } else {
                radix_trees[id_patch]->compute_cell_ibounding_box(
                    shamsys::instance::get_compute_queue());
                radix_trees[id_patch]->convert_bounding_box(shamsys::instance::get_compute_queue());
            }
        });

        shamsys::instance::get_compute_queue().wait();

        auto box = sched.get_box_tranform<vec3>();
        SimulationDomain<flt> sd(Free, std::get<0>(box), std::get<1>(box));

        // generate tree fields

        std::unordered_map<u64, std::unique_ptr<RadixTreeField<flt>>> cell_lengths;
        std::unordered_map<u64, std::unique_ptr<RadixTreeField<vec3>>> cell_centers;

        sched.for_each_patch_data([&](u64 id_patch, Patch & /*cur_p*/, PatchDataLayer &pdat) {
            auto &rtree = *radix_trees[id_patch];

            auto &c_len = cell_lengths[id_patch];
            auto &c_cen = cell_centers[id_patch];

            c_len = std::make_unique<RadixTreeField<flt>>();
            c_cen = std::make_unique<RadixTreeField<vec3>>();

            c_len->nvar = 1;
            c_cen->nvar = 1;

            auto &cell_length  = c_len->radix_tree_field_buf;
            auto &cell_centers = c_cen->radix_tree_field_buf;

            cell_centers = std::make_unique<sycl::buffer<vec3>>(
                rtree.tree_struct.internal_cell_count
                + rtree.tree_reduced_morton_codes.tree_leaf_count);
            cell_length = std::make_unique<sycl::buffer<flt>>(
                rtree.tree_struct.internal_cell_count
                + rtree.tree_reduced_morton_codes.tree_leaf_count);

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::range<1> range_tree = sycl::range<1>{
                    rtree.tree_reduced_morton_codes.tree_leaf_count
                    + rtree.tree_struct.internal_cell_count};

                auto pos_min_cell = sycl::accessor{
                    *rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
                auto pos_max_cell = sycl::accessor{
                    *rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};

                auto c_centers
                    = sycl::accessor{*cell_centers, cgh, sycl::write_only, sycl::no_init};
                auto c_length = sycl::accessor{*cell_length, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                    vec3 cur_pos_min_cell_a = pos_min_cell[item];
                    vec3 cur_pos_max_cell_a = pos_max_cell[item];

                    vec3 sa = (cur_pos_min_cell_a + cur_pos_max_cell_a) / 2;

                    vec3 dc_a = (cur_pos_max_cell_a - cur_pos_min_cell_a);

                    flt l_cell_a = sycl::max(sycl::max(dc_a.x(), dc_a.y()), dc_a.z());

                    c_centers[item] = sa;
                    c_length[item]  = l_cell_a;
                });
            });
        });

        std::unordered_map<u64, std::unique_ptr<RadixTreeField<flt>>> multipoles;

        sched.for_each_patch_data([&](u64 id_patch, Patch & /*cur_p*/, PatchDataLayer &pdat) {
            auto &rtree = *radix_trees[id_patch];

            u32 num_component_multipoles_fmm
                = (rtree.tree_struct.internal_cell_count
                   + rtree.tree_reduced_morton_codes.tree_leaf_count)
                  * SymTensorCollection<flt, 0, fmm_order>::num_component;

            auto &ref_field = multipoles[id_patch];
            ref_field       = std::make_unique<RadixTreeField<flt>>();

            ref_field->nvar       = SymTensorCollection<flt, 0, fmm_order>::num_component;
            auto &grav_multipoles = ref_field->radix_tree_field_buf;

            grav_multipoles = std::make_unique<sycl::buffer<flt>>(num_component_multipoles_fmm);

            compute_multipoles(
                rtree, pdat.get_field<vec3>(ixyz).get_buf(), *grav_multipoles, gpart_mass);
        });

        // generate the tree field for the box size info

        FullTreeField<flt, RadTree> min_slength;
        FullTreeField<flt, RadTree> max_slength;

        legacy::PatchField<flt> &max_slength_cells = max_slength.patch_field;
        legacy::PatchField<flt> &min_slength_cells = min_slength.patch_field;

        std::unordered_map<u64, flt> min_slength_map;
        std::unordered_map<u64, flt> max_slength_map;

        using RtreeField = typename RadTree::template RadixTreeField<flt>;
        std::unordered_map<u64, std::unique_ptr<RtreeField>> &min_tree_slength_map
            = min_slength.patch_tree_fields;
        std::unordered_map<u64, std::unique_ptr<RtreeField>> &max_tree_slength_map
            = max_slength.patch_tree_fields;

        for (auto &[k, rtree_ptr] : radix_trees) {
            auto [min, max]    = rtree_ptr->get_min_max_cell_side_length();
            min_slength_map[k] = min;
            max_slength_map[k] = max;
        }

        sched.compute_patch_field(
            min_slength_cells,
            get_mpi_type<flt>(),
            [&](sycl::queue & /*queue*/, Patch &p, PatchDataLayer & /*pdat*/) {
                return min_slength_map[p.id_patch];
            });

        sched.compute_patch_field(
            max_slength_cells,
            get_mpi_type<flt>(),
            [&](sycl::queue & /*queue*/, Patch &p, PatchDataLayer & /*pdat*/) {
                return max_slength_map[p.id_patch];
            });

        for (auto &[k, rtree_ptr] : radix_trees) {
            std::unique_ptr<RtreeField> &field_min = min_tree_slength_map[k];
            std::unique_ptr<RtreeField> &field_max = max_tree_slength_map[k];

            u32 total_cell_bount = rtree_ptr->tree_struct.internal_cell_count
                                   + rtree_ptr->tree_reduced_morton_codes.tree_leaf_count;

            field_min                       = std::make_unique<RtreeField>();
            field_min->nvar                 = 1;
            field_min->radix_tree_field_buf = std::make_unique<sycl::buffer<flt>>(total_cell_bount);

            field_max                       = std::make_unique<RtreeField>();
            field_max->nvar                 = 1;
            field_max->radix_tree_field_buf = std::make_unique<sycl::buffer<flt>>(total_cell_bount);

            auto &buf_pos_min_cell_flt = rtree_ptr->tree_cell_ranges.buf_pos_min_cell_flt;
            auto &buf_pos_max_cell_flt = rtree_ptr->tree_cell_ranges.buf_pos_max_cell_flt;

            auto &rfield_buf_min = field_min->radix_tree_field_buf;
            auto &rfield_buf_max = field_max->radix_tree_field_buf;

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor box_min_cell{*buf_pos_min_cell_flt, cgh, sycl::read_only};
                sycl::accessor box_max_cell{*buf_pos_max_cell_flt, cgh, sycl::read_only};

                sycl::accessor s_lengh_min{*rfield_buf_min, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor s_lengh_max{*rfield_buf_max, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{total_cell_bount}, [=](sycl::item<1> item) {
                    vec3 bmin = box_min_cell[item];
                    vec3 bmax = box_max_cell[item];

                    vec3 sz = bmax - bmin;

                    s_lengh_min[item] = sycl::fmin(sycl::fmin(sz.x(), sz.y()), sz.z());
                    s_lengh_max[item] = sycl::fmax(sycl::fmax(sz.x(), sz.y()), sz.z());
                });
            });
        }

        // make interfaces
        flt open_crit          = 0.5;
        using InterfHndl       = Interfacehandler<Tree_Send, flt, RadTree>;
        InterfHndl interf_hndl = InterfHndl();
        interf_hndl.compute_interface_list(
            sched,
            sptree,
            sd,
            radix_trees,
            FMMInteract_cd<flt>(open_crit),
            min_slength,
            max_slength);
        interf_hndl.initial_fetch(sched);
        interf_hndl.comm_trees();

        auto interf_pdat       = interf_hndl.comm_pdat(sched);
        auto interf_multipoles = interf_hndl.comm_tree_field(sched, multipoles);

        shamsys::instance::get_compute_queue().wait();

        // force

        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            shamlog_debug_ln("Selfgrav", "summing self grav to patch :", cur_p.id_patch);

            auto &pos_part_f  = pdat.get_field<vec3>(ixyz);
            auto &buf_force_f = pdat.get_field<vec3>(iaxyz);

            auto &pos_part_b  = pos_part_f.get_buf();
            auto &buf_force_b = buf_force_f.get_buf();

            auto &pos_part  = pos_part_b;
            auto &buf_force = buf_force_b;

            auto &rtree = *radix_trees[id_patch];

            auto &c_len = cell_lengths[id_patch];
            auto &c_cen = cell_centers[id_patch];

            auto &cell_length  = c_len->radix_tree_field_buf;
            auto &cell_centers = c_cen->radix_tree_field_buf;

            auto &grav_multipoles_f = multipoles[id_patch];
            auto &grav_multipoles   = grav_multipoles_f->radix_tree_field_buf;

            sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            auto xyz  = pos_part.get_read_access(depends_list);
            auto fxyz = buf_force.get_write_access(depends_list);

            auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
                using vec = vec3;

                using Rta = walker::Radix_tree_accessor<u_morton, vec>;
                Rta tree_acc(rtree, cgh);

                auto c_centers = sycl::accessor{*cell_centers, cgh, sycl::read_only};
                auto c_length  = sycl::accessor{*cell_length, cgh, sycl::read_only};

                sycl::range<1> range_leaf
                    = sycl::range<1>{rtree.tree_reduced_morton_codes.tree_leaf_count};

                u32 leaf_offset = rtree.tree_struct.internal_cell_count;

                auto multipoles = sycl::accessor{*grav_multipoles, cgh, sycl::read_only};

                const flt m             = gpart_mass;
                const auto open_crit_sq = open_crit * open_crit;

                cgh.parallel_for(range_leaf, [=](sycl::item<1> item) {
                    u32 id_cell_a = (u32) item.get_id(0) + leaf_offset;

                    vec cur_pos_min_cell_a = tree_acc.pos_min_cell[id_cell_a];
                    vec cur_pos_max_cell_a = tree_acc.pos_max_cell[id_cell_a];

                    vec sa       = c_centers[id_cell_a];
                    flt l_cell_a = c_length[id_cell_a];

                    auto dM_k = SymTensorCollection<flt, 1, fmm_order + 1>::zeros();

                    // out << id_cell_a << "\n";
                    // #if false
                    walker::rtree_for_cell(
                        tree_acc,
                        [&tree_acc,
                         &cur_pos_min_cell_a,
                         &cur_pos_max_cell_a,
                         &sa,
                         &l_cell_a,
                         &c_centers,
                         &c_length,
                         &open_crit_sq](u32 id_cell_b) {
                            vec cur_pos_min_cell_b = tree_acc.pos_min_cell[id_cell_b];
                            vec cur_pos_max_cell_b = tree_acc.pos_max_cell[id_cell_b];

                            vec sb       = c_centers[id_cell_b];
                            vec r_fmm    = sb - sa;
                            flt l_cell_b = c_length[id_cell_b];

                            flt opening_angle_sq = (l_cell_a + l_cell_b) * (l_cell_a + l_cell_b)
                                                   / sycl::dot(r_fmm, r_fmm);

                            using namespace walker::interaction_crit;

                            return sph_cell_cell_crit(
                                       cur_pos_min_cell_a,
                                       cur_pos_max_cell_a,
                                       cur_pos_min_cell_b,
                                       cur_pos_max_cell_b,
                                       0,
                                       0)
                                   || (opening_angle_sq > open_crit_sq);
                        },
                        [&](u32 node_b) {
                            // vec sb = c_centers[node_b];
                            // vec r_fmm = sb-sa;
                            // flt l_cell_b = c_length[node_b];

                            walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a) {
                                vec x_i = xyz[id_a];
                                vec sum_fi{0, 0, 0};

                                walker::iter_object_in_cell(tree_acc, node_b, [&](u32 id_b) {
                                    if (id_a != id_b) {
                                        vec x_j = xyz[id_b];

                                        vec real_r = x_i - x_j;

                                        flt inv_r_n = sycl::rsqrt(sycl::dot(real_r, real_r));
                                        sum_fi -= m * real_r * (inv_r_n * inv_r_n * inv_r_n);
                                    }
                                });

                                fxyz[id_a] += sum_fi;
                            });
                            //}
                        },
                        [&](u32 node_b) {
                            vec sb    = c_centers[node_b];
                            vec r_fmm = sb - sa;

                            auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(
                                multipoles,
                                node_b * SymTensorCollection<flt, 0, fmm_order>::num_component);
                            auto D_n = shamphys::GreenFuncGravCartesian<flt, 1, fmm_order + 1>::
                                get_der_tensors(r_fmm);

                            dM_k += get_dM_mat(D_n, Q_n);
                        });

                    // #endif

                    walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a) {
                        auto ai = SymTensorCollection<flt, 0, fmm_order>::from_vec(xyz[id_a] - sa);

                        auto tensor_to_sycl = [](SymTensor3d_1<flt> a) {
                            return vec{a.v_0, a.v_1, a.v_2};
                        };

                        vec tmp{0, 0, 0};

                        tmp += tensor_to_sycl(dM_k.t1 * ai.t0);
                        tmp += tensor_to_sycl(dM_k.t2 * ai.t1);
                        tmp += tensor_to_sycl(dM_k.t3 * ai.t2);
                        if constexpr (fmm_order >= 3) {
                            tmp += tensor_to_sycl(dM_k.t4 * ai.t3);
                        }
                        if constexpr (fmm_order >= 4) {
                            tmp += tensor_to_sycl(dM_k.t5 * ai.t4);
                        }
                        fxyz[id_a] -= tmp;

                        // auto dphi_0 = tensor_to_sycl(dM_k.t1*ai.t0);
                        // auto dphi_1 = tensor_to_sycl(dM_k.t2*ai.t1);
                        // auto dphi_2 = tensor_to_sycl(dM_k.t3*ai.t2);
                        // auto dphi_3 = tensor_to_sycl(dM_k.t4*ai.t3);
                        // auto dphi_4 = tensor_to_sycl(dM_k.t5*ai.t4);
                        // fxyz[id_a]  += dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;
                    });
                });
            });

            pos_part.complete_event_state(e);
            buf_force.complete_event_state(e);
        });

        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            shamlog_debug_ln("Selfgrav", "summing interf self grav to patch :", cur_p.id_patch);

            auto &pos_part  = pdat.get_field<vec3>(ixyz).get_buf();
            auto &buf_force = pdat.get_field<vec3>(iaxyz).get_buf();

            auto &rtree_cur = *radix_trees[id_patch];

            auto &c_len = cell_lengths[id_patch];
            auto &c_cen = cell_centers[id_patch];

            auto &cur_cell_length  = c_len->radix_tree_field_buf;
            auto &cur_cell_centers = c_cen->radix_tree_field_buf;

            for (u32 interf_id = 0; interf_id < interf_pdat[id_patch].size(); interf_id++) {
                shamlog_debug_ln(
                    "SelfGrav",
                    "adding interface",
                    std::get<0>(interf_hndl.tree_recv_map[id_patch][interf_id]));

                auto &rtree_interf = *std::get<1>(interf_hndl.tree_recv_map[id_patch][interf_id]);

                auto &pdat_interf = *std::get<1>(interf_pdat[id_patch][interf_id]);

                auto &pos_part_interf = pdat_interf.template get_field<vec3>(ixyz).get_buf();

                auto &multipole_interf = *std::get<1>(interf_multipoles[id_patch][interf_id]);

                // compute interface cell info
                auto interf_cell_centers = std::make_unique<sycl::buffer<vec3>>(
                    rtree_interf.tree_struct.internal_cell_count
                    + rtree_interf.tree_reduced_morton_codes.tree_leaf_count);
                auto interf_cell_length = std::make_unique<sycl::buffer<flt>>(
                    rtree_interf.tree_struct.internal_cell_count
                    + rtree_interf.tree_reduced_morton_codes.tree_leaf_count);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::range<1> range_tree = sycl::range<1>{
                        rtree_interf.tree_reduced_morton_codes.tree_leaf_count
                        + rtree_interf.tree_struct.internal_cell_count};

                    auto pos_min_cell = sycl::accessor{
                        *rtree_interf.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
                    auto pos_max_cell = sycl::accessor{
                        *rtree_interf.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};

                    auto c_centers = sycl::accessor{
                        *interf_cell_centers, cgh, sycl::write_only, sycl::no_init};
                    auto c_length
                        = sycl::accessor{*interf_cell_length, cgh, sycl::write_only, sycl::no_init};

                    cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                        vec3 cur_pos_min_cell_a = pos_min_cell[item];
                        vec3 cur_pos_max_cell_a = pos_max_cell[item];

                        vec3 sa = (cur_pos_min_cell_a + cur_pos_max_cell_a) / 2;

                        vec3 dc_a = (cur_pos_max_cell_a - cur_pos_min_cell_a);

                        flt l_cell_a = sycl::max(sycl::max(dc_a.x(), dc_a.y()), dc_a.z());

                        c_centers[item] = sa;
                        c_length[item]  = l_cell_a;
                    });
                });

                sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();
                sham::EventList depends_list;

                auto xyz        = pos_part.get_read_access(depends_list);
                auto xyz_interf = pos_part_interf.get_read_access(depends_list);
                auto fxyz       = buf_force.get_write_access(depends_list);

                auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
                    using vec = vec3;

                    using Rta = walker::Radix_tree_accessor<u_morton, vec>;
                    Rta tree_acc_curr(rtree_cur, cgh);
                    Rta tree_acc_interf(rtree_interf, cgh);

                    auto interf_c_centers
                        = sycl::accessor{*interf_cell_centers, cgh, sycl::read_only};
                    auto interf_c_length
                        = sycl::accessor{*interf_cell_length, cgh, sycl::read_only};

                    auto cur_c_centers = sycl::accessor{*cur_cell_centers, cgh, sycl::read_only};
                    auto cur_c_length  = sycl::accessor{*cur_cell_length, cgh, sycl::read_only};

                    sycl::range<1> cur_range_leaf
                        = sycl::range<1>{rtree_cur.tree_reduced_morton_codes.tree_leaf_count};

                    u32 cur_leaf_offset    = rtree_cur.tree_struct.internal_cell_count;
                    u32 interf_leaf_offset = rtree_interf.tree_struct.internal_cell_count;

                    auto multipoles = sycl::accessor{
                        *multipole_interf.radix_tree_field_buf, cgh, sycl::read_only};

                    const flt m = gpart_mass;

                    const auto open_crit_sq = open_crit * open_crit;

                    auto out = sycl::stream(4096, 4096, cgh);

                    cgh.parallel_for(cur_range_leaf, [=](sycl::item<1> item) {
                        u32 id_cell_a = (u32) item.get_id(0) + cur_leaf_offset;

                        vec cur_pos_min_cell_a = tree_acc_curr.pos_min_cell[id_cell_a];
                        vec cur_pos_max_cell_a = tree_acc_curr.pos_max_cell[id_cell_a];

                        vec sa       = cur_c_centers[id_cell_a];
                        flt l_cell_a = cur_c_length[id_cell_a];

                        auto dM_k = SymTensorCollection<flt, 1, fmm_order + 1>::zeros();

                        walker::rtree_for_cell(
                            tree_acc_interf,
                            [&](u32 id_cell_b) {
                                vec cur_pos_min_cell_b = tree_acc_interf.pos_min_cell[id_cell_b];
                                vec cur_pos_max_cell_b = tree_acc_interf.pos_max_cell[id_cell_b];

                                vec sb       = interf_c_centers[id_cell_b];
                                vec r_fmm    = sb - sa;
                                flt l_cell_b = interf_c_length[id_cell_b];

                                flt opening_angle_sq = (l_cell_a + l_cell_b) * (l_cell_a + l_cell_b)
                                                       / sycl::dot(r_fmm, r_fmm);

                                using namespace walker::interaction_crit;

                                return sph_cell_cell_crit(
                                           cur_pos_min_cell_a,
                                           cur_pos_max_cell_a,
                                           cur_pos_min_cell_b,
                                           cur_pos_max_cell_b,
                                           0,
                                           0)
                                       || (opening_angle_sq > open_crit_sq);
                            },
                            [&](u32 node_b) {
                                walker::iter_object_in_cell(
                                    tree_acc_curr, id_cell_a, [&](u32 id_a) {
                                        vec x_i = xyz[id_a];
                                        vec sum_fi{0, 0, 0};

                                        walker::iter_object_in_cell(
                                            tree_acc_interf, node_b, [&](u32 id_b) {
                                                // if(id_a != id_b){
                                                vec x_j = xyz_interf[id_b];

                                                vec real_r = x_i - x_j;

                                                flt inv_r_n
                                                    = sycl::rsqrt(sycl::dot(real_r, real_r));
                                                sum_fi
                                                    -= m * real_r * (inv_r_n * inv_r_n * inv_r_n);
                                                //}
                                            });

                                        // out << r_min << "\n";

                                        fxyz[id_a] += sum_fi;
                                    });
                                //}
                            },
                            [&](u32 node_b) {
                                vec sb = interf_c_centers[node_b];

                                vec r_fmm = sb - sa;

                                auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(
                                    multipoles,
                                    node_b * SymTensorCollection<flt, 0, fmm_order>::num_component);
                                auto D_n = shamphys::GreenFuncGravCartesian<flt, 1, fmm_order + 1>::
                                    get_der_tensors(r_fmm);

                                dM_k += get_dM_mat(D_n, Q_n);
                            });

                        walker::iter_object_in_cell(tree_acc_curr, id_cell_a, [&](u32 id_a) {
                            auto ai
                                = SymTensorCollection<flt, 0, fmm_order>::from_vec(xyz[id_a] - sa);

                            auto tensor_to_sycl = [](SymTensor3d_1<flt> a) {
                                return vec{a.v_0, a.v_1, a.v_2};
                            };

                            vec tmp{0, 0, 0};

                            tmp += tensor_to_sycl(dM_k.t1 * ai.t0);
                            tmp += tensor_to_sycl(dM_k.t2 * ai.t1);
                            tmp += tensor_to_sycl(dM_k.t3 * ai.t2);
                            if constexpr (fmm_order >= 3) {
                                tmp += tensor_to_sycl(dM_k.t4 * ai.t3);
                            }
                            if constexpr (fmm_order >= 4) {
                                tmp += tensor_to_sycl(dM_k.t5 * ai.t4);
                            }

                            fxyz[id_a] -= tmp;

                            // auto dphi_0 = tensor_to_sycl(dM_k.t1*ai.t0);
                            // auto dphi_1 = tensor_to_sycl(dM_k.t2*ai.t1);
                            // auto dphi_2 = tensor_to_sycl(dM_k.t3*ai.t2);
                            // auto dphi_3 = tensor_to_sycl(dM_k.t4*ai.t3);
                            // auto dphi_4 = tensor_to_sycl(dM_k.t5*ai.t4);
                            // fxyz[id_a]  += dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;
                        });
                    });
                });

                pos_part.complete_event_state(e);
                pos_part_interf.complete_event_state(e);
                buf_force.complete_event_state(e);

                shamsys::instance::get_compute_queue().wait();
            }
        });

        // leapfrog predictor
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            shamlog_debug_ln("SPHLeapfrog", "patch : n°", id_patch, "->", "corrector");

            lambda_correct(
                shamsys::instance::get_compute_scheduler().get_queue(),
                pdat,
                sycl::range<1>{pdat.get_obj_cnt()},
                dt_cur / 2);
        });

        return step_time;
    };

    f64 step_time = leapfrog_lambda(old_time, true, true);

    return step_time;
}

template<class flt>
void models::nbody::Nbody_SelfGrav<flt>::dump(std::string prefix) {
    std::cout << "dump : " << prefix << std::endl;
}

template<class flt>
void models::nbody::Nbody_SelfGrav<flt>::restart_dump(std::string prefix) {
    std::cout << "restart dump : " << prefix << std::endl;
}

template<class flt>
void models::nbody::Nbody_SelfGrav<flt>::close() {}

template class models::nbody::Nbody_SelfGrav<f32>;
