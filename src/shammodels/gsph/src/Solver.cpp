// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Guo
 * @brief GSPH Solver implementation
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 *
 * This implementation follows:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
 *   Godunov-type particle hydrodynamics"
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/Solver.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/UpdateDerivs.hpp"
#include "shammodels/gsph/modules/io/VTKDump.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include "shamtree/TreeTraversal.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::init_solver_graph() {

    storage.part_counts
        = std::make_shared<shamrock::solvergraph::Indexes<u32>>("part_counts", "N_{\\rm part}");

    storage.part_counts_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        "part_counts_with_ghost", "N_{\\rm part, with ghost}");

    storage.patch_rank_owner
        = std::make_shared<shamrock::solvergraph::ScalarsEdge<u32>>("patch_rank_owner", "rank");

    // Merged ghost spans
    storage.positions_with_ghosts
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("part_pos", "\\mathbf{r}");
    storage.hpart_with_ghosts
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_part", "h");

    storage.neigh_cache
        = std::make_shared<shammodels::sph::solvergraph::NeighCache>("neigh_cache", "neigh");

    storage.omega = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "omega", "\\Omega");
    storage.density = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "density", "\\rho");
    storage.pressure = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "pressure", "P");
    storage.soundspeed
        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "soundspeed", "c_s");
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::vtk_do_dump(
    std::string filename, bool add_patch_world_id) {

    modules::VTKDump<Tvec, Kern>(context, solver_config).do_dump(filename, add_patch_world_id);
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::gen_serial_patch_tree() {
    StackEntry stack_loc{};

    SerialPatchTree<Tvec> _sptree = SerialPatchTree<Tvec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::gen_ghost_handler(Tscal time_val) {
    StackEntry stack_loc{};

    using CfgClass = sph::BasicSPHGhostHandlerConfig<Tvec>;
    using BCConfig = typename CfgClass::Variant;

    using BCFree     = typename CfgClass::Free;
    using BCPeriodic = typename CfgClass::Periodic;
    using BCWall     = typename CfgClass::Wall;

    using SolverConfigBC   = typename Config::BCConfig;
    using SolverBCFree     = typename SolverConfigBC::Free;
    using SolverBCPeriodic = typename SolverConfigBC::Periodic;
    using SolverBCWall     = typename SolverConfigBC::Wall;

    // Boundary condition selection - similar to SPH solver
    if (SolverBCFree *c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
        storage.ghost_handler.set(
            GhostHandle{scheduler(), BCFree{}, storage.patch_rank_owner});
    } else if (
        SolverBCPeriodic *c
        = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
        storage.ghost_handler.set(
            GhostHandle{scheduler(), BCPeriodic{}, storage.patch_rank_owner});
    } else if (SolverBCWall *c = std::get_if<SolverBCWall>(&solver_config.boundary_config.config)) {
        // For wall boundaries, use Wall ghost handling
        // Wall particles created manually provide boundary neighbor support
        storage.ghost_handler.set(
            GhostHandle{scheduler(), BCWall{}, storage.patch_rank_owner});
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::build_ghost_cache() {
    StackEntry stack_loc{};

    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;
    SPHUtils sph_utils(scheduler());

    storage.ghost_patch_cache.set(sph_utils.build_interf_cache(
        storage.ghost_handler.get(),
        storage.serial_patch_tree.get(),
        solver_config.htol_up_coarse_cycle));
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::clear_ghost_cache() {
    StackEntry stack_loc{};
    storage.ghost_patch_cache.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::merge_position_ghost() {
    StackEntry stack_loc{};

    storage.merged_xyzh.set(
        storage.ghost_handler.get().build_comm_merge_positions(storage.ghost_patch_cache.get()));

    // Set element counts
    shambase::get_check_ref(storage.part_counts).indexes
        = storage.merged_xyzh.get().template map<u32>(
            [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                return scheduler().patch_data.get_pdat(id).get_obj_cnt();
            });

    // Set element counts with ghost
    shambase::get_check_ref(storage.part_counts_with_ghost).indexes
        = storage.merged_xyzh.get().template map<u32>(
            [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                return mpdat.get_obj_cnt();
            });

    // Attach spans to block coords
    shambase::get_check_ref(storage.positions_with_ghosts)
        .set_refs(storage.merged_xyzh.get()
                      .template map<std::reference_wrapper<PatchDataField<Tvec>>>(
                          [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                              return std::ref(mpdat.get_field<Tvec>(0));
                          }));

    shambase::get_check_ref(storage.hpart_with_ghosts)
        .set_refs(storage.merged_xyzh.get()
                      .template map<std::reference_wrapper<PatchDataField<Tscal>>>(
                          [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                              return std::ref(mpdat.get_field<Tscal>(1));
                          }));
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::build_merged_pos_trees() {
    StackEntry stack_loc{};

    auto &merged_xyzh = storage.merged_xyzh.get();
    auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

    shambase::DistributedData<RTree> trees
        = merged_xyzh.template map<RTree>([&](u64 id, shamrock::patch::PatchDataLayer &merged) {
              PatchDataField<Tvec> &pos = merged.template get_field<Tvec>(0);
              Tvec bmax                 = pos.compute_max();
              Tvec bmin                 = pos.compute_min();

              shammath::AABB<Tvec> aabb(bmin, bmax);

              Tscal infty = std::numeric_limits<Tscal>::infinity();

              // Ensure that no particle is on the boundary of the AABB
              aabb.lower[0] = std::nextafter(aabb.lower[0], -infty);
              aabb.lower[1] = std::nextafter(aabb.lower[1], -infty);
              aabb.lower[2] = std::nextafter(aabb.lower[2], -infty);
              aabb.upper[0] = std::nextafter(aabb.upper[0], infty);
              aabb.upper[1] = std::nextafter(aabb.upper[1], infty);
              aabb.upper[2] = std::nextafter(aabb.upper[2], infty);

              auto bvh = RTree::make_empty(dev_sched);
              bvh.rebuild_from_positions(
                  pos.get_buf(), pos.get_obj_cnt(), aabb, solver_config.tree_reduction_level);

              return bvh;
          });

    storage.merged_pos_trees.set(std::move(trees));
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::clear_merged_pos_trees() {
    StackEntry stack_loc{};
    storage.merged_pos_trees.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::compute_presteps_rint() {
    StackEntry stack_loc{};

    auto &xyzh_merged = storage.merged_xyzh.get();
    auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

    storage.rtree_rint_field.set(
        storage.merged_pos_trees.get().template map<shamtree::KarrasRadixTreeField<Tscal>>(
            [&](u64 id, RTree &rtree) -> shamtree::KarrasRadixTreeField<Tscal> {
                shamrock::patch::PatchDataLayer &tmp = xyzh_merged.get(id);
                auto &buf                            = tmp.get_field_buf_ref<Tscal>(1);
                auto buf_int = shamtree::new_empty_karras_radix_tree_field<Tscal>();

                auto ret = shamtree::compute_tree_field_max_field<Tscal>(
                    rtree.structure,
                    rtree.reduced_morton_set.get_leaf_cell_iterator(),
                    std::move(buf_int),
                    buf);

                // Increase the size by tolerance factor
                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{},
                    sham::MultiRef{ret.buf_field},
                    ret.buf_field.get_size(),
                    [htol = solver_config.htol_up_coarse_cycle](u32 i, Tscal *h_tree) {
                        h_tree[i] *= htol;
                    });

                return std::move(ret);
            }));
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_presteps_rint() {
    storage.rtree_rint_field.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::start_neighbors_cache() {
    StackEntry stack_loc{};

    shambase::Timer time_neigh;
    time_neigh.start();

    Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

    // Build neighbor cache using tree traversal - same approach as SPH module
    auto build_neigh_cache = [&](u64 patch_id) -> shamrock::tree::ObjectCache {
        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        // Allocate neighbor count buffer
        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamsys::instance::get_compute_queue().wait_and_throw();

        // First pass: count neighbors
        {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            auto xyz             = buf_xyz.get_read_access(depends_list);
            auto hpart           = buf_hpart.get_read_access(depends_list);
            auto rint_tree       = tree_field_rint.get_read_access(depends_list);
            auto neigh_cnt       = neigh_count.get_write_access(depends_list);
            auto particle_looper = obj_it.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&, h_tolerance](sycl::handler &cgh) {
                shambase::parallel_for(cgh, obj_cnt, "gsph_count_neighbors", [=](u64 gid) {
                    u32 id_a = (u32)gid;

                    Tscal rint_a = hpart[id_a] * h_tolerance;
                    Tvec xyz_a = xyz[id_a];

                    Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                    Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                    u32 cnt = 0;

                    particle_looper.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(
                                xyz_a,
                                inter_box_a_min,
                                inter_box_a_max,
                                node_aabb.lower,
                                node_aabb.upper,
                                int_r_max_cell);
                        },
                        [&](u32 id_b) {
                            Tvec dr = xyz_a - xyz[id_b];
                            Tscal rab2 = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact
                                = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            cnt += (no_interact) ? 0 : 1;
                        });

                    neigh_cnt[id_a] = cnt;
                });
            });

            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            neigh_count.complete_event_state(e);
            tree_field_rint.complete_event_state(e);
            obj_it.complete_event_state(e);
        }

        // Use tree::prepare_object_cache to do prefix sum and allocate buffers
        shamrock::tree::ObjectCache pcache = shamrock::tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        // Second pass: fill neighbor indices
        {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            auto xyz               = buf_xyz.get_read_access(depends_list);
            auto hpart             = buf_hpart.get_read_access(depends_list);
            auto rint_tree         = tree_field_rint.get_read_access(depends_list);
            auto scanned_neigh_cnt = pcache.scanned_cnt.get_read_access(depends_list);
            auto neigh             = pcache.index_neigh_map.get_write_access(depends_list);
            auto particle_looper   = obj_it.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&, h_tolerance](sycl::handler &cgh) {
                shambase::parallel_for(cgh, obj_cnt, "gsph_fill_neighbors", [=](u64 gid) {
                    u32 id_a = (u32)gid;

                    Tscal rint_a = hpart[id_a] * h_tolerance;
                    Tvec xyz_a = xyz[id_a];

                    Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                    Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                    u32 write_idx = scanned_neigh_cnt[id_a];

                    particle_looper.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(
                                xyz_a,
                                inter_box_a_min,
                                inter_box_a_max,
                                node_aabb.lower,
                                node_aabb.upper,
                                int_r_max_cell);
                        },
                        [&](u32 id_b) {
                            Tvec dr = xyz_a - xyz[id_b];
                            Tscal rab2 = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact
                                = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            if (!no_interact) {
                                neigh[write_idx++] = id_b;
                            }
                        });
                });
            });

            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            tree_field_rint.complete_event_state(e);
            pcache.scanned_cnt.complete_event_state(e);
            pcache.index_neigh_map.complete_event_state(e);
            obj_it.complete_event_state(e);
        }

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.end();
    storage.timings_details.neighbors += time_neigh.elasped_sec();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_neighbors_cache() {
    storage.neigh_cache->neigh_cache = {};
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::gsph_prestep(Tscal time_val, Tscal dt) {
    StackEntry stack_loc{};

    shamlog_debug_ln("GSPH", "Prestep at t =", time_val, "dt =", dt);
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::apply_position_boundary(Tscal time_val) {
    StackEntry stack_loc{};

    shamlog_debug_ln("GSPH", "apply position boundary");

    PatchScheduler &sched = scheduler();
    shamrock::SchedulerUtility integrators(sched);
    shamrock::ReattributeDataUtility reatrib(sched);

    auto &pdl = sched.pdl();
    const u32 ixyz    = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz   = pdl.get_field_idx<Tvec>("vxyz");
    auto [bmin, bmax] = sched.get_box_volume<Tvec>();

    using SolverConfigBC   = typename Config::BCConfig;
    using SolverBCFree     = typename SolverConfigBC::Free;
    using SolverBCPeriodic = typename SolverConfigBC::Periodic;

    if (SolverBCFree *c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
        if (shamcomm::world_rank() == 0) {
            logger::info_ln("PositionUpdated", "free boundaries skipping geometry update");
        }
    } else if (
        SolverBCPeriodic *c
        = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
        integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});
    }

    reatrib.reatribute_patch_objects(storage.serial_patch_tree.get(), "xyz");
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::do_predictor_leapfrog(Tscal dt) {
    StackEntry stack_loc{};
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");
    const u32 iwall_flag      = pdl.get_field_idx<u32>("wall_flag");

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
    const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

    Tscal half_dt = dt / 2;

    // Predictor step with wall particle skip
    // Wall particles (wall_flag != 0) are boundary particles and should not be time integrated
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0) return;

        auto &xyz_field       = pdat.get_field<Tvec>(ixyz);
        auto &vxyz_field      = pdat.get_field<Tvec>(ivxyz);
        auto &axyz_field      = pdat.get_field<Tvec>(iaxyz);
        auto &wall_flag_field = pdat.get_field<u32>(iwall_flag);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        // Forward euler: v += a*dt/2, x += v*dt, v += a*dt/2 (leapfrog kick-drift-kick)
        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{axyz_field.get_buf(), wall_flag_field.get_buf()},
            sham::MultiRef{xyz_field.get_buf(), vxyz_field.get_buf()},
            cnt,
            [half_dt, dt](u32 i, const Tvec *axyz, const u32 *wall_flag, Tvec *xyz, Tvec *vxyz) {
                // Skip wall particles (not time integrated)
                if (wall_flag[i] != 0) return;

                // Kick: v += a*dt/2
                vxyz[i] += axyz[i] * half_dt;
                // Drift: x += v*dt
                xyz[i] += vxyz[i] * dt;
                // Kick: v += a*dt/2
                vxyz[i] += axyz[i] * half_dt;
            });

        // Internal energy integration (if adiabatic EOS)
        if (has_uint) {
            auto &uint_field  = pdat.get_field<Tscal>(iuint);
            auto &duint_field = pdat.get_field<Tscal>(iduint);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{duint_field.get_buf(), wall_flag_field.get_buf()},
                sham::MultiRef{uint_field.get_buf()},
                cnt,
                [half_dt, dt](u32 i, const Tscal *duint, const u32 *wall_flag, Tscal *uint) {
                    // Skip wall particles
                    if (wall_flag[i] != 0) return;

                    // u += du*dt/2 + du*dt/2 = du*dt
                    uint[i] += duint[i] * dt;
                });
        }
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::init_ghost_layout() {
    StackEntry stack_loc{};

    // Reset first in case it was set from a previous timestep
    storage.ghost_layout.reset();
    storage.ghost_layout.set(std::make_shared<shamrock::patch::PatchDataLayerLayout>());

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());

    solver_config.set_ghost_layout(ghost_layout);
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::communicate_merge_ghosts_fields() {
    StackEntry stack_loc{};

    shambase::Timer timer_interf;
    timer_interf.start();

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
    const u32 iwall_flag      = pdl.get_field_idx<u32>("wall_flag");

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

    auto ghost_layout_ptr                               = storage.ghost_layout.get();
    shamrock::patch::PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(ghost_layout_ptr);
    u32 ihpart_interf     = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 ivxyz_interf      = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf     = ghost_layout.get_field_idx<Tscal>("omega");
    u32 idensity_interf   = ghost_layout.get_field_idx<Tscal>("density");
    u32 iwall_flag_interf = ghost_layout.get_field_idx<u32>("wall_flag");
    u32 iuint_interf      = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;

    using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
    shamrock::solvergraph::Field<Tscal> &omega    = shambase::get_check_ref(storage.omega);
    shamrock::solvergraph::Field<Tscal> &density  = shambase::get_check_ref(storage.density);

    // Build interface data from ghost cache
    auto pdat_interf = ghost_handle.template build_interface_native<PatchDataLayer>(
        storage.ghost_patch_cache.get(),
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sham::DeviceBuffer<u32> &buf_idx, u32 cnt) {
            PatchDataLayer pdat(ghost_layout_ptr);
            pdat.reserve(cnt);
            return pdat;
        });

    // Populate interface data with field values
    ghost_handle.template modify_interface_native<PatchDataLayer>(
        storage.ghost_patch_cache.get(),
        pdat_interf,
        [&](u64 sender,
            u64,
            InterfaceBuildInfos binfo,
            sham::DeviceBuffer<u32> &buf_idx,
            u32 cnt,
            PatchDataLayer &pdat) {
            PatchDataLayer &sender_patch          = scheduler().patch_data.get_pdat(sender);
            PatchDataField<Tscal> &sender_omega   = omega.get(sender);
            PatchDataField<Tscal> &sender_density = density.get(sender);

            sender_patch.get_field<Tscal>(ihpart).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(ihpart_interf));
            sender_patch.get_field<Tvec>(ivxyz).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tvec>(ivxyz_interf));
            sender_omega.append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(iomega_interf));
            sender_density.append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(idensity_interf));
            sender_patch.get_field<u32>(iwall_flag).append_subset_to(
                buf_idx, cnt, pdat.get_field<u32>(iwall_flag_interf));

            if (has_uint) {
                sender_patch.get_field<Tscal>(iuint).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(iuint_interf));
            }
        });

    // Apply velocity offset for periodic boundaries
    ghost_handle.template modify_interface_native<PatchDataLayer>(
        storage.ghost_patch_cache.get(),
        pdat_interf,
        [&](u64 sender,
            u64,
            InterfaceBuildInfos binfo,
            sham::DeviceBuffer<u32> &buf_idx,
            u32 cnt,
            PatchDataLayer &pdat) {
            if (sycl::length(binfo.offset_speed) > 0) {
                pdat.get_field<Tvec>(ivxyz_interf).apply_offset(binfo.offset_speed);
            }
        });

    // Communicate ghost data across MPI ranks
    shambase::DistributedDataShared<PatchDataLayer> interf_pdat
        = ghost_handle.communicate_pdat(ghost_layout_ptr, std::move(pdat_interf));

    // Count total ghost particles per patch
    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchDataLayer &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    // Merge local and ghost data
    storage.merged_patchdata_ghost.set(
        ghost_handle.template merge_native<PatchDataLayer, PatchDataLayer>(
            std::move(interf_pdat),
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                PatchDataLayer pdat_new(ghost_layout_ptr);

                u32 or_elem = pdat.get_obj_cnt();
                pdat_new.reserve(or_elem + sz_interf_map[p.id_patch]);

                PatchDataField<Tscal> &cur_omega   = omega.get(p.id_patch);
                PatchDataField<Tscal> &cur_density = density.get(p.id_patch);

                // Insert local particle data
                pdat_new.get_field<Tscal>(ihpart_interf).insert(pdat.get_field<Tscal>(ihpart));
                pdat_new.get_field<Tvec>(ivxyz_interf).insert(pdat.get_field<Tvec>(ivxyz));
                pdat_new.get_field<Tscal>(iomega_interf).insert(cur_omega);
                pdat_new.get_field<Tscal>(idensity_interf).insert(cur_density);
                pdat_new.get_field<u32>(iwall_flag_interf).insert(pdat.get_field<u32>(iwall_flag));

                if (has_uint) {
                    pdat_new.get_field<Tscal>(iuint_interf).insert(pdat.get_field<Tscal>(iuint));
                }

                pdat_new.check_field_obj_cnt_match();
                return pdat_new;
            },
            [](PatchDataLayer &pdat, PatchDataLayer &pdat_interf) {
                pdat.insert_elements(pdat_interf);
            }));

    timer_interf.end();
    storage.timings_details.interface += timer_interf.elasped_sec();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_merge_ghosts_fields() {
    storage.merged_patchdata_ghost.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::compute_omega() {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    const Tscal pmass = solver_config.gpart_mass;

    // Verify particle mass is valid
    if (shamcomm::world_rank() == 0) {
        if (pmass <= Tscal(0) || pmass < Tscal(1e-100) || !std::isfinite(pmass)) {
            logger::warn_ln("GSPH", "Invalid particle mass in compute_omega: pmass =", pmass);
        }
    }

    shamrock::solvergraph::Field<Tscal> &omega_field = shambase::get_check_ref(storage.omega);
    shamrock::solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);

    // Create sizes directly from scheduler to ensure we have all patches
    shambase::DistributedData<u32> sizes;
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        sizes.add_obj(p.id_patch, pdat.get_obj_cnt());
    });

    // Ensure fields are allocated for all patches with correct sizes
    omega_field.ensure_sizes(sizes);
    density_field.ensure_sizes(sizes);

    // Compute density and omega via SPH summation using neighbor cache
    // Reference: g_pre_interaction.cpp from sphcode
    // dens_i = sum_j m_j * W(r_ij, h_i)
    // dh_dens_i = sum_j m_j * dhW(r_ij, h_i)
    // omega = 1 / (1 + h/(D*rho) * dh_rho)

    auto &merged_xyzh = storage.merged_xyzh.get();
    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;
    constexpr u32 DIM = 3;  // 3D

    // Configuration for h iteration
    constexpr u32 MAX_H_ITER = 30;      // Maximum iterations for h convergence
    constexpr Tscal H_TOL = Tscal(1e-3); // Convergence tolerance for h
    constexpr Tscal H_EVOL_MAX = Tscal(1.2); // Max h change per iteration

    // Get patchdata layout for hpart field
    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0) return;

        // Get merged position/h buffer (includes ghosts for neighbor lookup)
        auto &mfield = merged_xyzh.get(p.id_patch);
        sham::DeviceBuffer<Tvec> &buf_xyz = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart_merged = mfield.template get_field_buf_ref<Tscal>(1);

        // Get patchdata hpart field for writing (this is the source of truth)
        sham::DeviceBuffer<Tscal> &buf_hpart_local = pdat.get_field_buf_ref<Tscal>(ihpart);

        auto &omega_buf = omega_field.get_field(p.id_patch).get_buf();
        auto &density_buf = density_field.get_field(p.id_patch).get_buf();

        // Get neighbor cache
        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(p.id_patch);

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto xyz = buf_xyz.get_read_access(depends_list);
        auto hpart_merged = buf_hpart_merged.get_read_access(depends_list);
        auto hpart_local = buf_hpart_local.get_write_access(depends_list);
        auto omega = omega_buf.get_write_access(depends_list);
        auto density = density_buf.get_write_access(depends_list);
        auto ploop_ptrs = pcache.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            shambase::parallel_for(cgh, cnt, "compute_density_omega_with_h_iter", [=](u64 gid) {
                u32 id_a = (u32)gid;

                const Tvec xyz_a = xyz[id_a];
                Tscal h_a = hpart_merged[id_a];  // Start with current h

                // Newton-Raphson iteration to find h that satisfies rho_sum = rho_h(m, h)
                // rho_h = m * (hfact/h)^3
                // Target: find h such that SPH summation density matches rho_h
                for (u32 iter = 0; iter < MAX_H_ITER; iter++) {
                    const Tscal h_old = h_a;
                    const Tscal dint = h_a * h_a * Rker2;

                    // SPH density summation
                    Tscal rho_sum = Tscal(0);
                    Tscal sumdWdh = Tscal(0);

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        const Tvec dr = xyz_a - xyz[id_b];
                        const Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 >= dint) {
                            return;
                        }

                        const Tscal rab = sycl::sqrt(rab2);

                        // SPH density summation: rho = sum_j m_j W(r_ij, h_i)
                        rho_sum += pmass * Kernel::W_3d(rab, h_a);

                        // Derivative w.r.t. h: dhW_3d (kernel provides this)
                        sumdWdh += pmass * Kernel::dhW_3d(rab, h_a);
                    });

                    // Target density from h: rho_h = m * (hfact/h)^3
                    const Tscal hfact = Kernel::hfactd;
                    const Tscal rho_h = pmass * (hfact / h_a) * (hfact / h_a) * (hfact / h_a);

                    // Newton-Raphson update for h
                    // f(h) = rho_sum - rho_h = 0
                    // df/dh = sumdWdh + 3*rho_h/h  (since drho_h/dh = -3*rho_h/h)
                    const Tscal f_val = rho_sum - rho_h;
                    const Tscal df_val = sumdWdh + Tscal(DIM) * rho_h / h_a;

                    if (sycl::fabs(df_val) > Tscal(1e-30)) {
                        Tscal new_h = h_a - f_val / df_val;

                        // Clamp h change per iteration
                        if (new_h < h_a / H_EVOL_MAX) {
                            new_h = h_a / H_EVOL_MAX;
                        }
                        if (new_h > h_a * H_EVOL_MAX) {
                            new_h = h_a * H_EVOL_MAX;
                        }

                        // Ensure h is positive
                        new_h = sycl::max(new_h, Tscal(1e-10));

                        h_a = new_h;
                    }

                    // Check convergence
                    if (sycl::fabs(h_a - h_old) / h_old < H_TOL) {
                        break;
                    }
                }

                // Final density computation with converged h
                const Tscal dint_final = h_a * h_a * Rker2;
                Tscal dens = Tscal(0);
                Tscal dh_dens = Tscal(0);

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    const Tvec dr = xyz_a - xyz[id_b];
                    const Tscal rab2 = sycl::dot(dr, dr);

                    if (rab2 >= dint_final) {
                        return;
                    }

                    const Tscal rab = sycl::sqrt(rab2);

                    dens += pmass * Kernel::W_3d(rab, h_a);

                    // For omega computation
                    Tscal dW_dr = Kernel::dW_3d(rab, h_a);
                    Tscal dhW = -dW_dr * rab / h_a - Tscal(DIM) * Kernel::W_3d(rab, h_a) / h_a;
                    dh_dens += pmass * dhW;
                });

                // Store updated h to patchdata
                hpart_local[id_a] = h_a;

                // Store density
                density[id_a] = dens;

                // Grad-h correction factor: omega = 1 / (1 + h/(D*rho) * dh_rho)
                Tscal omega_val = Tscal(1);
                if (dens > Tscal(1e-30)) {
                    omega_val = Tscal(1) / (Tscal(1) + h_a / (Tscal(DIM) * dens) * dh_dens);
                    omega_val = sycl::clamp(omega_val, Tscal(0.5), Tscal(1.5));
                }
                omega[id_a] = omega_val;
            });
        });

        buf_xyz.complete_event_state(e);
        buf_hpart_merged.complete_event_state(e);
        buf_hpart_local.complete_event_state(e);
        omega_buf.complete_event_state(e);
        density_buf.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::compute_eos_fields() {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    // GSPH EOS: Following reference implementation (g_pre_interaction.cpp)
    // P = (γ - 1) * ρ * u  where ρ is from SPH summation
    // c = sqrt(γ * (γ - 1) * u)  -- from internal energy, not from P/ρ

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    const Tscal gamma = solver_config.gamma;
    const bool has_uint = solver_config.has_field_uint();

    // Get ghost layout field indices
    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");
    u32 iuint_interf    = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;

    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);
    shamrock::solvergraph::Field<Tscal> &soundspeed_field = shambase::get_check_ref(storage.soundspeed);

    // Size buffers to part_counts_with_ghost (includes ghosts!)
    shambase::DistributedData<u32> &counts_with_ghosts
        = shambase::get_check_ref(storage.part_counts_with_ghost).indexes;

    pressure_field.ensure_sizes(counts_with_ghosts);
    soundspeed_field.ensure_sizes(counts_with_ghosts);

    // Iterate over merged_patchdata_ghost (includes local + ghost particles)
    storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
        u32 total_elements = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);
        if (total_elements == 0) return;

        // Use SPH-summation density from communicated ghost data
        sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);
        auto &pressure_buf = pressure_field.get_field(id).get_buf();
        auto &soundspeed_buf = soundspeed_field.get_field(id).get_buf();

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto density = buf_density.get_read_access(depends_list);
        auto pressure = pressure_buf.get_write_access(depends_list);
        auto soundspeed = soundspeed_buf.get_write_access(depends_list);

        const Tscal *uint_ptr = nullptr;
        if (has_uint) {
            uint_ptr = mpdat.get_field_buf_ref<Tscal>(iuint_interf).get_read_access(depends_list);
        }

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parallel_for(cgh, total_elements, "compute_eos_gsph", [=](u64 gid) {
                u32 i = (u32)gid;

                // Use SPH-summation density (from compute_omega, communicated to ghosts)
                Tscal rho = density[i];
                rho = sycl::max(rho, Tscal(1e-30));

                if (has_uint && uint_ptr != nullptr) {
                    // Adiabatic EOS (reference: g_pre_interaction.cpp line 107)
                    // P = (γ - 1) * ρ * u
                    Tscal u = uint_ptr[i];
                    u = sycl::max(u, Tscal(1e-30));
                    Tscal P = (gamma - Tscal(1.0)) * rho * u;

                    // Sound speed from internal energy (reference: solver.cpp line 2661)
                    // c = sqrt(γ * (γ - 1) * u)
                    Tscal cs = sycl::sqrt(gamma * (gamma - Tscal(1.0)) * u);

                    // Clamp to reasonable values
                    P = sycl::clamp(P, Tscal(1e-30), Tscal(1e30));
                    cs = sycl::clamp(cs, Tscal(1e-10), Tscal(1e10));

                    pressure[i] = P;
                    soundspeed[i] = cs;
                } else {
                    // Isothermal case
                    Tscal cs = Tscal(1.0);
                    Tscal P = cs * cs * rho;

                    pressure[i] = P;
                    soundspeed[i] = cs;
                }
            });
        });

        // Complete all buffer event states
        buf_density.complete_event_state(e);
        if (has_uint) {
            mpdat.get_field_buf_ref<Tscal>(iuint_interf).complete_event_state(e);
        }
        pressure_buf.complete_event_state(e);
        soundspeed_buf.complete_event_state(e);
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_eos_fields() {
    // Reset computed EOS fields - they're recomputed each timestep
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::prepare_corrector() {
    StackEntry stack_loc{};

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 iaxyz = pdl.get_field_idx<Tvec>("axyz");

    // Create compute field to store old acceleration
    auto old_axyz = utility.make_compute_field<Tvec>("old_axyz", 1);

    // Copy current acceleration to old_axyz
    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0) return;

        auto &axyz_field     = pdat.get_field<Tvec>(iaxyz);
        auto &old_axyz_field = old_axyz.get_field(p.id_patch);

        // Copy using kernel_call
        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{axyz_field.get_buf()},
            sham::MultiRef{old_axyz_field.get_buf()},
            cnt,
            [](u32 i, const Tvec *src, Tvec *dst) {
                dst[i] = src[i];
            });
    });

    storage.old_axyz.set(std::move(old_axyz));

    if (solver_config.has_field_uint()) {
        const u32 iduint = pdl.get_field_idx<Tscal>("duint");
        auto old_duint   = utility.make_compute_field<Tscal>("old_duint", 1);

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0) return;

            auto &duint_field     = pdat.get_field<Tscal>(iduint);
            auto &old_duint_field = old_duint.get_field(p.id_patch);

            // Copy using kernel_call
            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{duint_field.get_buf()},
                sham::MultiRef{old_duint_field.get_buf()},
                cnt,
                [](u32 i, const Tscal *src, Tscal *dst) {
                    dst[i] = src[i];
                });
        });

        storage.old_duint.set(std::move(old_duint));
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::update_derivs() {
    StackEntry stack_loc{};
    // GSPH derivative update using Riemann solver
    gsph::modules::UpdateDerivs<Tvec, Kern>(context, solver_config, storage).update_derivs();
}

template<class Tvec, template<class> class Kern>
typename shammodels::gsph::Solver<Tvec, Kern>::Tscal
shammodels::gsph::Solver<Tvec, Kern>::compute_dt_cfl() {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
    const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");

    shamrock::solvergraph::Field<Tscal> &soundspeed_field
        = shambase::get_check_ref(storage.soundspeed);

    Tscal C_cour  = solver_config.cfl_config.cfl_cour;
    Tscal C_force = solver_config.cfl_config.cfl_force;

    // Use ComputeField for proper reduction support
    shamrock::SchedulerUtility utility(scheduler());
    ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", 1);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0) return;

        auto &buf_hpart  = pdat.get_field_buf_ref<Tscal>(ihpart);
        auto &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        auto &buf_cs     = soundspeed_field.get_field(cur_p.id_patch).get_buf();
        auto &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto hpart      = buf_hpart.get_read_access(depends_list);
        auto axyz       = buf_axyz.get_read_access(depends_list);
        auto cs         = buf_cs.get_read_access(depends_list);
        auto cfl_dt_acc = cfl_dt_buf.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parallel_for(cgh, cnt, "gsph_compute_cfl_dt", [=](u64 gid) {
                u32 i = (u32) gid;

                Tscal h_i   = hpart[i];
                Tscal cs_i  = cs[i];
                Tscal abs_a = sycl::length(axyz[i]);

                // Guard against invalid values (NaN/Inf)
                if (!sycl::isfinite(h_i) || h_i <= Tscal(0)) h_i = Tscal(1e-10);
                if (!sycl::isfinite(cs_i) || cs_i <= Tscal(0)) cs_i = Tscal(1e-10);
                if (!sycl::isfinite(abs_a)) abs_a = Tscal(1e30);

                // Sound CFL condition: dt = C_cour * h / c_s
                // Following Kitajima et al. (2025) simple form for GSPH
                Tscal dt_c = C_cour * h_i / cs_i;

                // Force condition: dt = C_force * sqrt(h / |a|)
                Tscal dt_f = C_force * sycl::sqrt(h_i / (abs_a + Tscal(1e-30)));

                Tscal dt_min = sycl::min(dt_c, dt_f);

                // Ensure a valid finite timestep with minimum floor
                if (!sycl::isfinite(dt_min) || dt_min <= Tscal(0)) {
                    dt_min = Tscal(1e-10);  // Minimum timestep floor
                }

                cfl_dt_acc[i] = dt_min;
            });
        });

        buf_hpart.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_cs.complete_event_state(e);
        cfl_dt_buf.complete_event_state(e);
    });

    // Compute minimum across all patches on this rank
    Tscal rank_dt = cfl_dt.compute_rank_min();

    // Guard against invalid reduction result
    if (!std::isfinite(rank_dt) || rank_dt <= Tscal(0)) {
        rank_dt = Tscal(1e-6);  // Reasonable floor for SPH simulations
    }

    // Global reduction across MPI ranks
    Tscal global_min_dt = shamalgs::collective::allreduce_min(rank_dt);

    // Final safety floor to prevent simulation stalling
    // For typical SPH simulations, timestep should be O(h/cs) ~ O(1e-4)
    // Use 1e-6 as minimum floor to prevent extreme stalling
    const Tscal dt_min_floor = Tscal(1e-6);
    if (!std::isfinite(global_min_dt) || global_min_dt < dt_min_floor) {
        global_min_dt = dt_min_floor;
    }

    return global_min_dt;
}

template<class Tvec, template<class> class Kern>
bool shammodels::gsph::Solver<Tvec, Kern>::apply_corrector(Tscal dt, u64 Npart_all) {
    StackEntry stack_loc{};

    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");
    const u32 iwall_flag = pdl.get_field_idx<u32>("wall_flag");

    Tscal half_dt = Tscal{0.5} * dt;

    // Corrector: v = v + 0.5*(a_new - a_old)*dt
    // Skip wall particles (not time integrated)
    scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0) return;

        auto &vxyz       = pdat.get_field<Tvec>(ivxyz);
        auto &axyz       = pdat.get_field<Tvec>(iaxyz);
        auto &wall_flag  = pdat.get_field<u32>(iwall_flag);
        auto &old_axyz   = storage.old_axyz.get().get_field(p.id_patch);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{axyz.get_buf(), old_axyz.get_buf(), wall_flag.get_buf()},
            sham::MultiRef{vxyz.get_buf()},
            cnt,
            [half_dt](u32 i, const Tvec *axyz_new, const Tvec *axyz_old, const u32 *wall_flag, Tvec *vxyz) {
                // Skip wall particles
                if (wall_flag[i] != 0) return;
                vxyz[i] += half_dt * (axyz_new[i] - axyz_old[i]);
            });
    });

    if (solver_config.has_field_uint()) {
        const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
        const u32 iduint = pdl.get_field_idx<Tscal>("duint");

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0) return;

            auto &uint_field = pdat.get_field<Tscal>(iuint);
            auto &duint      = pdat.get_field<Tscal>(iduint);
            auto &wall_flag  = pdat.get_field<u32>(iwall_flag);
            auto &old_duint  = storage.old_duint.get().get_field(p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{duint.get_buf(), old_duint.get_buf(), wall_flag.get_buf()},
                sham::MultiRef{uint_field.get_buf()},
                cnt,
                [half_dt](u32 i, const Tscal *duint_new, const Tscal *duint_old, const u32 *wall_flag, Tscal *uint) {
                    // Skip wall particles
                    if (wall_flag[i] != 0) return;
                    uint[i] += half_dt * (duint_new[i] - duint_old[i]);
                });
        });

        storage.old_duint.reset();
    }

    storage.old_axyz.reset();

    return true;
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::update_sync_load_values() {
    // Update load balancing values - simplified for now
}

template<class Tvec, template<class> class Kern>
shammodels::gsph::TimestepLog shammodels::gsph::Solver<Tvec, Kern>::evolve_once() {

    // Validate configuration before running
    solver_config.check_config_runtime();

    Tscal t_current = solver_config.get_time();
    Tscal dt        = solver_config.get_dt();

    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        shamcomm::logs::raw_ln(
            shambase::format("---------------- GSPH t = {}, dt = {} ----------------", t_current, dt));
    }

    shambase::Timer tstep;
    tstep.start();

    // Load balancing step
    scheduler().scheduler_step(true, true);
    scheduler().scheduler_step(false, false);

    // Give to the solvergraph the patch rank owners
    storage.patch_rank_owner->values = {};
    scheduler().for_each_global_patch([&](const shamrock::patch::Patch p) {
        storage.patch_rank_owner->values.add_obj(
            p.id_patch, scheduler().get_patch_rank_owner(p.id_patch));
    });

    using namespace shamrock;
    using namespace shamrock::patch;

    u64 Npart_all = scheduler().get_total_obj_count();

    // =========================================================================
    // CORRECTED SIMULATION LOOP ORDER (matching reference SPH code)
    // =========================================================================
    // The key insight from the reference code is that density/EOS must be
    // computed AFTER the predictor step, on the NEW positions. Otherwise,
    // the forces are computed using stale EOS values.
    //
    // Loop order:
    // 1. PREDICTOR: move particles using OLD accelerations
    // 2. BOUNDARY: apply periodic/free boundary conditions
    // 3. TREE BUILD: build spatial trees on NEW positions
    // 4. DENSITY/EOS: compute density, pressure, soundspeed on NEW positions
    // 5. FORCES: compute accelerations using FRESH EOS
    // 6. CORRECTOR: refine velocities using average of old/new accelerations
    // 7. CFL: compute next timestep
    // =========================================================================

    // STEP 1: PREDICTOR - move particles using OLD accelerations
    // (On first iteration, accelerations are zero, so this is just position drift)
    do_predictor_leapfrog(dt);

    // STEP 2: BOUNDARY - apply boundary conditions to NEW positions
    // Build serial patch tree first (needed for boundary application)
    gen_serial_patch_tree();
    apply_position_boundary(t_current + dt);

    // STEP 3: TREE BUILD - build trees on NEW positions
    // Generate ghost handler for the new positions
    gen_ghost_handler(t_current + dt);

    // Build ghost cache for interface exchange
    build_ghost_cache();

    // Merge positions with ghosts
    merge_position_ghost();

    // Build trees over merged positions
    build_merged_pos_trees();

    // Compute interaction ranges
    compute_presteps_rint();

    // Build neighbor cache
    start_neighbors_cache();

    // STEP 4: DENSITY/OMEGA - compute on NEW positions
    // Compute omega (grad-h correction factor) - needed for force computation
    compute_omega();

    // Initialize ghost layout BEFORE communication
    init_ghost_layout();

    // Communicate ghost fields (hpart, uint, vxyz, omega)
    // This MUST happen BEFORE compute_eos_fields so EOS can be computed for ghosts
    communicate_merge_ghosts_fields();

    // STEP 4b: EOS - compute AFTER ghost communication (CRITICAL!)
    // This ensures P and cs are computed for ALL particles (local + ghost)
    // Following SPH pattern: EOS is computed on merged_patchdata_ghost
    compute_eos_fields();

    // STEP 5: FORCES - compute accelerations using FRESH EOS
    // Save old accelerations for corrector
    prepare_corrector();

    // Update derivatives using GSPH Riemann solver
    update_derivs();

    // STEP 6: CORRECTOR - refine velocities
    apply_corrector(dt, Npart_all);

    // STEP 7: CFL - compute next timestep
    Tscal dt_next = compute_dt_cfl();

    // Ensure dt doesn't grow too fast (max 2x per step), but allow any value if dt was 0
    if (dt > Tscal(0)) {
        dt_next = sham::min(dt_next, Tscal(2) * dt);
    }

    // Cleanup for next iteration
    reset_neighbors_cache();
    reset_presteps_rint();
    clear_merged_pos_trees();
    reset_merge_ghosts_fields();
    storage.merged_xyzh.reset();
    clear_ghost_cache();
    reset_serial_patch_tree();
    reset_ghost_handler();
    storage.ghost_layout.reset();

    // Update time
    solver_config.set_time(t_current + dt);
    solver_config.set_next_dt(dt_next);

    solve_logs.step_count++;

    tstep.end();

    // Prepare timing log
    TimestepLog log;
    log.rank     = shamcomm::world_rank();
    log.rate     = Tscal(Npart_all) / tstep.elasped_sec();
    log.npart    = Npart_all;
    log.tcompute = tstep.elasped_sec();

    return log;
}

// Template instantiations
using namespace shammath;

// M-spline kernels (Monaghan)
template class shammodels::gsph::Solver<f64_3, M4>;
template class shammodels::gsph::Solver<f64_3, M6>;
template class shammodels::gsph::Solver<f64_3, M8>;

// Wendland kernels (C2, C4, C6) - recommended for GSPH (Inutsuka 2002)
template class shammodels::gsph::Solver<f64_3, C2>;
template class shammodels::gsph::Solver<f64_3, C4>;
template class shammodels::gsph::Solver<f64_3, C6>;
