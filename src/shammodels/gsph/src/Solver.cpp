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
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
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
#include "shammodels/gsph/math/sr/primitive_recovery.hpp"
#include "shammodels/gsph/modules/UpdateDerivs.hpp"
#include "shammodels/gsph/modules/io/VTKDump.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/modules/LoopSmoothingLengthIter.hpp"
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

    storage.omega    = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "omega", "\\Omega");
    storage.density  = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "density", "\\rho");
    storage.pressure = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "pressure", "P");
    storage.soundspeed
        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "soundspeed", "c_s");

    // Initialize gradient fields for MUSCL reconstruction
    // These are only used when reconstruct_config.is_muscl() == true
    storage.grad_density
        = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_density", "\\nabla\\rho");
    storage.grad_pressure
        = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_pressure", "\\nabla P");
    storage.grad_vx
        = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_vx", "\\nabla v_x");
    storage.grad_vy
        = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_vy", "\\nabla v_y");
    storage.grad_vz
        = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "grad_vz", "\\nabla v_z");
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

    using BCFree             = typename CfgClass::Free;
    using BCPeriodic         = typename CfgClass::Periodic;
    using BCShearingPeriodic = typename CfgClass::ShearingPeriodic;

    using SolverConfigBC           = typename Config::BCConfig;
    using SolverBCFree             = typename SolverConfigBC::Free;
    using SolverBCPeriodic         = typename SolverConfigBC::Periodic;
    using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;

    // Boundary condition selection - similar to SPH solver
    // Note: Wall boundaries use Periodic with dynamic wall particles
    if (SolverBCFree *c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
        storage.ghost_handler.set(
            GhostHandle{
                scheduler(), BCFree{}, storage.patch_rank_owner, storage.xyzh_ghost_layout});
    } else if (
        SolverBCPeriodic *c
        = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
        storage.ghost_handler.set(
            GhostHandle{
                scheduler(), BCPeriodic{}, storage.patch_rank_owner, storage.xyzh_ghost_layout});
    } else if (
        SolverBCShearingPeriodic *c
        = std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)) {
        // Shearing periodic boundaries (Stone 2010) - reuse SPH implementation
        storage.ghost_handler.set(
            GhostHandle{
                scheduler(),
                BCShearingPeriodic{
                    c->shear_base, c->shear_dir, c->shear_speed * time_val, c->shear_speed},
                storage.patch_rank_owner,
                storage.xyzh_ghost_layout});
    } else {
        shambase::throw_with_loc<std::runtime_error>("GSPH: Unsupported boundary condition type.");
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
        .set_refs(
            storage.merged_xyzh.get().template map<std::reference_wrapper<PatchDataField<Tvec>>>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tvec>(0));
                }));

    shambase::get_check_ref(storage.hpart_with_ghosts)
        .set_refs(
            storage.merged_xyzh.get().template map<std::reference_wrapper<PatchDataField<Tscal>>>(
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
                    u32 id_a = (u32) gid;

                    Tscal rint_a = hpart[id_a] * h_tolerance;
                    Tvec xyz_a   = xyz[id_a];

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
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
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
        shamrock::tree::ObjectCache pcache
            = shamrock::tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

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
                    u32 id_a = (u32) gid;

                    Tscal rint_a = hpart[id_a] * h_tolerance;
                    Tvec xyz_a   = xyz[id_a];

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
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
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

    auto &pdl         = sched.pdl();
    const u32 ixyz    = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz   = pdl.get_field_idx<Tvec>("vxyz");
    auto [bmin, bmax] = sched.get_box_volume<Tvec>();

    using SolverConfigBC           = typename Config::BCConfig;
    using SolverBCFree             = typename SolverConfigBC::Free;
    using SolverBCPeriodic         = typename SolverConfigBC::Periodic;
    using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;

    if (SolverBCFree *c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
        if (shamcomm::world_rank() == 0) {
            logger::info_ln("PositionUpdated", "free boundaries skipping geometry update");
        }
    } else if (
        SolverBCPeriodic *c
        = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
        integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});
    } else if (
        SolverBCShearingPeriodic *c
        = std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)) {
        // Apply shearing periodic boundaries (Stone 2010) - reuse SPH implementation
        integrators.fields_apply_shearing_periodicity(
            ixyz,
            ivxyz,
            std::pair{bmin, bmax},
            c->shear_base,
            c->shear_dir,
            c->shear_speed * time_val,
            c->shear_speed);
    } else {
        shambase::throw_with_loc<std::runtime_error>("GSPH: Unsupported boundary condition type.");
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

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
    const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

    Tscal half_dt = dt / 2;

    // Predictor step: leapfrog kick-drift-kick
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &xyz_field  = pdat.get_field<Tvec>(ixyz);
        auto &vxyz_field = pdat.get_field<Tvec>(ivxyz);
        auto &axyz_field = pdat.get_field<Tvec>(iaxyz);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        // Leapfrog KDK: first half-kick, then drift
        // The second half-kick (corrector) happens AFTER force recomputation
        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{axyz_field.get_buf()},
            sham::MultiRef{xyz_field.get_buf(), vxyz_field.get_buf()},
            cnt,
            [half_dt, dt](u32 i, const Tvec *axyz, Tvec *xyz, Tvec *vxyz) {
                // First kick: v += a*dt/2 (using OLD acceleration)
                vxyz[i] += axyz[i] * half_dt;
                // Drift: x += v*dt
                xyz[i] += vxyz[i] * dt;
            });

        // Internal energy integration (if adiabatic EOS)
        // Predictor: u += du*dt/2 (first half-step)
        // The second half-step happens in the corrector after force recomputation
        if (has_uint) {
            auto &uint_field  = pdat.get_field<Tscal>(iuint);
            auto &duint_field = pdat.get_field<Tscal>(iduint);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{duint_field.get_buf()},
                sham::MultiRef{uint_field.get_buf()},
                cnt,
                [half_dt](u32 i, const Tscal *duint, Tscal *uint) {
                    // u += du*dt/2 (first half-step)
                    uint[i] += duint[i] * half_dt;
                });
        }
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::init_ghost_layout() {
    StackEntry stack_loc{};

    // Initialize xyzh_ghost_layout for BasicSPHGhostHandler (position + smoothing length)
    storage.xyzh_ghost_layout = std::make_shared<shamrock::patch::PatchDataLayerLayout>();
    storage.xyzh_ghost_layout->template add_field<Tvec>("xyz", 1);
    storage.xyzh_ghost_layout->template add_field<Tscal>("hpart", 1);

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

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

    auto ghost_layout_ptr                               = storage.ghost_layout.get();
    shamrock::patch::PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(ghost_layout_ptr);
    u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>("omega");
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");
    u32 iuint_interf    = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;

    // Gradient field indices (for MUSCL reconstruction)
    const bool has_grads = solver_config.requires_gradients();
    u32 igrad_d_interf   = has_grads ? ghost_layout.get_field_idx<Tvec>("grad_density") : 0;
    u32 igrad_p_interf   = has_grads ? ghost_layout.get_field_idx<Tvec>("grad_pressure") : 0;
    u32 igrad_vx_interf  = has_grads ? ghost_layout.get_field_idx<Tvec>("grad_vx") : 0;
    u32 igrad_vy_interf  = has_grads ? ghost_layout.get_field_idx<Tvec>("grad_vy") : 0;
    u32 igrad_vz_interf  = has_grads ? ghost_layout.get_field_idx<Tvec>("grad_vz") : 0;

    using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
    shamrock::solvergraph::Field<Tscal> &omega    = shambase::get_check_ref(storage.omega);
    shamrock::solvergraph::Field<Tscal> &density  = shambase::get_check_ref(storage.density);

    // Get gradient fields (for MUSCL)
    shamrock::solvergraph::Field<Tvec> *grad_density_ptr
        = has_grads ? &shambase::get_check_ref(storage.grad_density) : nullptr;
    shamrock::solvergraph::Field<Tvec> *grad_pressure_ptr
        = has_grads ? &shambase::get_check_ref(storage.grad_pressure) : nullptr;
    shamrock::solvergraph::Field<Tvec> *grad_vx_ptr
        = has_grads ? &shambase::get_check_ref(storage.grad_vx) : nullptr;
    shamrock::solvergraph::Field<Tvec> *grad_vy_ptr
        = has_grads ? &shambase::get_check_ref(storage.grad_vy) : nullptr;
    shamrock::solvergraph::Field<Tvec> *grad_vz_ptr
        = has_grads ? &shambase::get_check_ref(storage.grad_vz) : nullptr;

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

            if (has_uint) {
                sender_patch.get_field<Tscal>(iuint).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(iuint_interf));
            }

            // Communicate gradient fields for MUSCL reconstruction
            if (has_grads) {
                grad_density_ptr->get(sender).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(igrad_d_interf));
                grad_pressure_ptr->get(sender).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(igrad_p_interf));
                grad_vx_ptr->get(sender).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(igrad_vx_interf));
                grad_vy_ptr->get(sender).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(igrad_vy_interf));
                grad_vz_ptr->get(sender).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(igrad_vz_interf));
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

                if (has_uint) {
                    pdat_new.get_field<Tscal>(iuint_interf).insert(pdat.get_field<Tscal>(iuint));
                }

                // Insert local gradient data for MUSCL reconstruction
                if (has_grads) {
                    pdat_new.get_field<Tvec>(igrad_d_interf)
                        .insert(grad_density_ptr->get(p.id_patch));
                    pdat_new.get_field<Tvec>(igrad_p_interf)
                        .insert(grad_pressure_ptr->get(p.id_patch));
                    pdat_new.get_field<Tvec>(igrad_vx_interf).insert(grad_vx_ptr->get(p.id_patch));
                    pdat_new.get_field<Tvec>(igrad_vy_interf).insert(grad_vy_ptr->get(p.id_patch));
                    pdat_new.get_field<Tvec>(igrad_vz_interf).insert(grad_vz_ptr->get(p.id_patch));
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

    const Tscal pmass = solver_config.gpart_mass;

    // Verify particle mass is valid
    if (shamcomm::world_rank() == 0) {
        if (pmass <= Tscal(0) || pmass < Tscal(1e-100) || !std::isfinite(pmass)) {
            logger::warn_ln("GSPH", "Invalid particle mass in compute_omega: pmass =", pmass);
        }
    }

    shamrock::solvergraph::Field<Tscal> &omega_field   = shambase::get_check_ref(storage.omega);
    shamrock::solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);

    // Create sizes directly from scheduler to ensure we have all patches
    std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes
        = std::make_shared<shamrock::solvergraph::Indexes<u32>>("sizes", "N");
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
    });

    // Ensure fields are allocated for all patches with correct sizes
    omega_field.ensure_sizes(sizes->indexes);
    density_field.ensure_sizes(sizes->indexes);

    // Get patchdata layout for hpart field
    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");

    // =========================================================================
    // OUTER-LOOP SMOOTHING LENGTH ITERATION (FIX FOR CACHE CONSISTENCY BUG)
    // =========================================================================
    // The original implementation had an inner-loop Newton-Raphson iteration
    // inside a GPU kernel. This caused issues because:
    // 1. Neighbor cache was built with OLD h values (+ 10% tolerance)
    // 2. Inner iteration could change h by more than 10%
    // 3. Particles that should be neighbors weren't found in the cache
    // 4. Result: underestimated density at discontinuities -> wrong forces
    //
    // The fix uses the SPH-style outer-loop approach:
    // 1. Create GSPH IterateSmoothingLengthDensity module (ONE step per call)
    // 2. Wrap in LoopSmoothingLengthIter for multiple iterations
    // 3. If h grows beyond tolerance, signal for cache rebuild
    // =========================================================================

    auto &merged_xyzh = storage.merged_xyzh.get();

    // Create field references for the iteration module
    // Position spans (from merged xyzh)
    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> pos_merged
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("pos", "r");
    shamrock::solvergraph::DDPatchDataFieldRef<Tvec> pos_refs = {};

    // Old h spans (from merged xyzh - read only during iteration)
    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hold
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_old", "h^{old}");
    shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hold_refs = {};

    // New h spans (local patchdata - written during iteration)
    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hnew
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_new", "h^{new}");
    shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};

    // Populate field references
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        auto &mfield = merged_xyzh.get(p.id_patch);

        // Position from merged data (includes ghosts for neighbor search)
        pos_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tvec>(0)));

        // h_old from merged data
        hold_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tscal>(1)));

        // h_new to local patchdata (this is updated during iteration)
        hnew_refs.add_obj(p.id_patch, std::ref(pdat.get_field<Tscal>(ihpart)));
    });

    pos_merged->set_refs(pos_refs);
    hold->set_refs(hold_refs);
    hnew->set_refs(hnew_refs);

    // Initialize hnew with hold values
    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &mfield           = merged_xyzh.get(p.id_patch);
        auto &buf_hpart_merged = mfield.template get_field_buf_ref<Tscal>(1);
        auto &buf_hpart_local  = pdat.get_field_buf_ref<Tscal>(ihpart);

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{buf_hpart_merged},
            sham::MultiRef{buf_hpart_local},
            cnt,
            [](u32 i, const Tscal *h_old, Tscal *h_new) {
                h_new[i] = h_old[i];
            });
    });

    // Create epsilon field for convergence tracking
    shamrock::SchedulerUtility utility(scheduler());
    ComputeField<Tscal> _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1);

    // Initialize epsilon to large value (not converged)
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &eps_buf = _epsilon_h.get_buf_check(p.id_patch);

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{},
            sham::MultiRef{eps_buf},
            cnt,
            [](u32 i, Tscal *eps) {
                eps[i] = Tscal(1.0); // Start with large epsilon
            });
    });

    // Create epsilon field references
    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> eps_h
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("eps_h", "\\epsilon_h");
    shamrock::solvergraph::DDPatchDataFieldRef<Tscal> eps_h_refs = {};
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        auto &field = _epsilon_h.get_field(p.id_patch);
        eps_h_refs.add_obj(p.id_patch, std::ref(field));
    });
    eps_h->set_refs(eps_h_refs);

    // Use SPH's IterateSmoothingLengthDensity module (reuse, no duplication)
    std::shared_ptr<sph::modules::IterateSmoothingLengthDensity<Tvec, Kernel>> smth_h_iter
        = std::make_shared<sph::modules::IterateSmoothingLengthDensity<Tvec, Kernel>>(
            solver_config.gpart_mass,
            solver_config.htol_up_coarse_cycle,
            solver_config.htol_up_fine_cycle);

    // SPH's module only iterates h, no density/omega outputs
    smth_h_iter->set_edges(sizes, storage.neigh_cache, pos_merged, hold, hnew, eps_h);

    // Create convergence flag
    std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> is_converged
        = std::make_shared<shamrock::solvergraph::ScalarEdge<bool>>("is_converged", "converged");

    // Use LoopSmoothingLengthIter from SPH module for outer loop iteration
    shammodels::sph::modules::LoopSmoothingLengthIter<Tvec> loop_smth_h_iter(
        smth_h_iter, solver_config.epsilon_h, solver_config.h_iter_per_subcycles, false);
    loop_smth_h_iter.set_edges(eps_h, is_converged);

    // Run the outer loop iteration
    loop_smth_h_iter.evaluate();

    // Check convergence
    if (!is_converged->value) {
        // Get convergence statistics
        Tscal local_max_eps  = shamrock::solvergraph::get_rank_max(*eps_h);
        Tscal global_max_eps = shamalgs::collective::allreduce_max(local_max_eps);

        // Count particles that need cache rebuild (eps == -1)
        u64 cnt_unconverged = 0;
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto res = _epsilon_h.get_field(p.id_patch).get_ids_buf_where([](auto access, u32 id) {
                return access[id] < Tscal(0);
            });
            cnt_unconverged += std::get<1>(res);
        });
        u64 global_cnt_unconverged = shamalgs::collective::allreduce_sum(cnt_unconverged);

        if (shamcomm::world_rank() == 0) {
            if (global_cnt_unconverged > 0) {
                logger::warn_ln(
                    "GSPH",
                    "Smoothing length iteration: ",
                    global_cnt_unconverged,
                    " particles need cache rebuild (h grew beyond tolerance)");
            } else {
                logger::warn_ln(
                    "GSPH",
                    "Smoothing length iteration did not converge, max eps =",
                    global_max_eps);
            }
        }
    }

    // =========================================================================
    // COMPUTE DENSITY AND OMEGA AFTER H CONVERGENCE
    // =========================================================================
    // Now that h has converged, compute the final density and omega values.
    // This is done ONCE here instead of on every iteration (more efficient).
    // =========================================================================

    static constexpr Tscal Rkern = Kernel::Rkern;

    auto &neigh_cache = storage.neigh_cache->neigh_cache;

    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &mfield = merged_xyzh.get(p.id_patch);
        auto &pcache = neigh_cache.get(p.id_patch);

        // Get position and h from merged data (includes ghosts for neighbor search)
        auto &buf_xyz   = mfield.template get_field_buf_ref<Tvec>(0);
        auto &buf_hpart = pdat.get_field_buf_ref<Tscal>(ihpart);

        // Get density and omega output fields
        auto &dens_field = density_field.get_field(p.id_patch);
        auto &omeg_field = omega_field.get_field(p.id_patch);

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto ploop_ptrs  = pcache.get_read_access(depends_list);
        auto xyz_acc     = buf_xyz.get_read_access(depends_list);
        auto h_acc       = buf_hpart.get_read_access(depends_list);
        auto density_acc = dens_field.get_buf().get_write_access(depends_list);
        auto omega_acc   = omeg_field.get_buf().get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            shambase::parallel_for(cgh, cnt, "gsph_compute_density_omega", [=](u64 gid) {
                u32 id_a = (u32) gid;

                Tvec xyz_a = xyz_acc[id_a];
                Tscal h_a  = h_acc[id_a];
                Tscal dint = h_a * h_a * Rkern * Rkern;

                // SPH density summation
                Tscal rho_sum = Tscal(0);
                Tscal sumdWdh = Tscal(0);

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    Tvec dr    = xyz_a - xyz_acc[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);

                    if (rab2 > dint) {
                        return;
                    }

                    Tscal rab = sycl::sqrt(rab2);

                    rho_sum += pmass * Kernel::W_3d(rab, h_a);
                    sumdWdh += pmass * Kernel::dhW_3d(rab, h_a);
                });

                // Store density
                density_acc[id_a] = sycl::max(rho_sum, Tscal(1e-30));

                // Compute omega (grad-h correction factor)
                // Omega = 1 + h/(dim*rho) * (drho/dh)
                // This matches SPH's ComputeOmega and is used in sph_pressure_symetric
                // which divides by (rho^2 * omega), so we need Omega not 1/Omega
                Tscal omega_val = Tscal(1);
                if (rho_sum > Tscal(1e-30)) {
                    omega_val = Tscal(1) + h_a / (Tscal(dim) * rho_sum) * sumdWdh;
                    omega_val = sycl::clamp(omega_val, Tscal(0.5), Tscal(2.0));
                }
                omega_acc[id_a] = omega_val;
            });
        });

        // Complete event states for all accessed buffers
        pcache.complete_event_state({e});
        buf_xyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        dens_field.get_buf().complete_event_state(e);
        omeg_field.get_buf().complete_event_state(e);
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

    auto dev_sched      = shamsys::instance::get_compute_scheduler_ptr();
    const Tscal gamma   = solver_config.get_eos_gamma();
    const bool has_uint = solver_config.has_field_uint();

    // Get ghost layout field indices
    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");
    u32 iuint_interf    = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;

    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);
    shamrock::solvergraph::Field<Tscal> &soundspeed_field
        = shambase::get_check_ref(storage.soundspeed);

    // Size buffers to part_counts_with_ghost (includes ghosts!)
    shambase::DistributedData<u32> &counts_with_ghosts
        = shambase::get_check_ref(storage.part_counts_with_ghost).indexes;

    pressure_field.ensure_sizes(counts_with_ghosts);
    soundspeed_field.ensure_sizes(counts_with_ghosts);

    // Iterate over merged_patchdata_ghost (includes local + ghost particles)
    storage.merged_patchdata_ghost.get().for_each([&](u64 id, PatchDataLayer &mpdat) {
        u32 total_elements
            = shambase::get_check_ref(storage.part_counts_with_ghost).indexes.get(id);
        if (total_elements == 0)
            return;

        // Use SPH-summation density from communicated ghost data
        sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);
        auto &pressure_buf                     = pressure_field.get_field(id).get_buf();
        auto &soundspeed_buf                   = soundspeed_field.get_field(id).get_buf();

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto density    = buf_density.get_read_access(depends_list);
        auto pressure   = pressure_buf.get_write_access(depends_list);
        auto soundspeed = soundspeed_buf.get_write_access(depends_list);

        const Tscal *uint_ptr = nullptr;
        if (has_uint) {
            uint_ptr = mpdat.get_field_buf_ref<Tscal>(iuint_interf).get_read_access(depends_list);
        }

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parallel_for(cgh, total_elements, "compute_eos_gsph", [=](u64 gid) {
                u32 i = (u32) gid;

                // Use SPH-summation density (from compute_omega, communicated to ghosts)
                Tscal rho = density[i];
                rho       = sycl::max(rho, Tscal(1e-30));

                if (has_uint && uint_ptr != nullptr) {
                    // Adiabatic EOS (reference: g_pre_interaction.cpp line 107)
                    // P = (γ - 1) * ρ * u
                    Tscal u = uint_ptr[i];
                    u       = sycl::max(u, Tscal(1e-30));
                    Tscal P = (gamma - Tscal(1.0)) * rho * u;

                    // Sound speed from internal energy (reference: solver.cpp line 2661)
                    // c = sqrt(γ * (γ - 1) * u)
                    Tscal cs = sycl::sqrt(gamma * (gamma - Tscal(1.0)) * u);

                    // Clamp to reasonable values
                    P  = sycl::clamp(P, Tscal(1e-30), Tscal(1e30));
                    cs = sycl::clamp(cs, Tscal(1e-10), Tscal(1e10));

                    pressure[i]   = P;
                    soundspeed[i] = cs;
                } else {
                    // Isothermal case
                    Tscal cs = Tscal(1.0);
                    Tscal P  = cs * cs * rho;

                    pressure[i]   = P;
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
void shammodels::gsph::Solver<Tvec, Kern>::compute_gradients() {
    StackEntry stack_loc{};

    // Only compute gradients for MUSCL reconstruction
    if (!solver_config.requires_gradients()) {
        return;
    }

    using namespace shamrock;
    using namespace shamrock::patch;

    const Tscal pmass = solver_config.gpart_mass;
    const Tscal gamma = solver_config.get_eos_gamma();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const bool has_uint       = solver_config.has_field_uint();
    const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

    // Get gradient fields from storage
    shamrock::solvergraph::Field<Tvec> &grad_density_field
        = shambase::get_check_ref(storage.grad_density);
    shamrock::solvergraph::Field<Tvec> &grad_pressure_field
        = shambase::get_check_ref(storage.grad_pressure);
    shamrock::solvergraph::Field<Tvec> &grad_vx_field = shambase::get_check_ref(storage.grad_vx);
    shamrock::solvergraph::Field<Tvec> &grad_vy_field = shambase::get_check_ref(storage.grad_vy);
    shamrock::solvergraph::Field<Tvec> &grad_vz_field = shambase::get_check_ref(storage.grad_vz);

    // Get density field for SPH-summation density
    shamrock::solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);

    // Ensure gradient fields have correct sizes
    shambase::DistributedData<u32> &counts = shambase::get_check_ref(storage.part_counts).indexes;

    grad_density_field.ensure_sizes(counts);
    grad_pressure_field.ensure_sizes(counts);
    grad_vx_field.ensure_sizes(counts);
    grad_vy_field.ensure_sizes(counts);
    grad_vz_field.ensure_sizes(counts);

    auto &merged_xyzh = storage.merged_xyzh.get();
    auto &neigh_cache = storage.neigh_cache->neigh_cache;

    static constexpr Tscal Rkern = Kernel::Rkern;

    // Compute gradients following reference implementation (g_pre_interaction.cpp)
    // grad_d = Σ_j m_j ∇W_ij
    // grad_p = (grad_d * u_i + du) * (γ - 1)  where du = Σ_j m_j (u_j - u_i) ∇W_ij
    // grad_v = Σ_j m_j (v_j - v_i) ∇W_ij / ρ_i
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &mfield = merged_xyzh.get(p.id_patch);
        auto &pcache = neigh_cache.get(p.id_patch);

        // Get position, h, velocity from merged data
        auto &buf_xyz   = mfield.template get_field_buf_ref<Tvec>(0);
        auto &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);
        auto &buf_vxyz  = pdat.get_field_buf_ref<Tvec>(ivxyz);

        // Get density (local particles only)
        auto &dens_field = density_field.get_field(p.id_patch);

        // Get gradient output fields
        auto &grad_d_field = grad_density_field.get_field(p.id_patch);
        auto &grad_p_field = grad_pressure_field.get_field(p.id_patch);
        auto &grad_vx_buf  = grad_vx_field.get_field(p.id_patch);
        auto &grad_vy_buf  = grad_vy_field.get_field(p.id_patch);
        auto &grad_vz_buf  = grad_vz_field.get_field(p.id_patch);

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto ploop_ptrs  = pcache.get_read_access(depends_list);
        auto xyz_acc     = buf_xyz.get_read_access(depends_list);
        auto h_acc       = buf_hpart.get_read_access(depends_list);
        auto v_acc       = buf_vxyz.get_read_access(depends_list);
        auto dens_acc    = dens_field.get_buf().get_read_access(depends_list);
        auto grad_d_acc  = grad_d_field.get_buf().get_write_access(depends_list);
        auto grad_p_acc  = grad_p_field.get_buf().get_write_access(depends_list);
        auto grad_vx_acc = grad_vx_buf.get_buf().get_write_access(depends_list);
        auto grad_vy_acc = grad_vy_buf.get_buf().get_write_access(depends_list);
        auto grad_vz_acc = grad_vz_buf.get_buf().get_write_access(depends_list);

        // Get internal energy if adiabatic
        const Tscal *uint_ptr = nullptr;
        if (has_uint) {
            uint_ptr = pdat.get_field_buf_ref<Tscal>(iuint).get_read_access(depends_list);
        }

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            shambase::parallel_for(cgh, cnt, "gsph_compute_gradients", [=](u64 gid) {
                u32 id_a = (u32) gid;

                Tvec xyz_a  = xyz_acc[id_a];
                Tscal h_a   = h_acc[id_a];
                Tvec v_a    = v_acc[id_a];
                Tscal rho_a = sycl::max(dens_acc[id_a], Tscal(1e-30));
                Tscal dint  = h_a * h_a * Rkern * Rkern;

                // Get internal energy for particle a
                Tscal u_a = Tscal(0);
                if (uint_ptr != nullptr) {
                    u_a = uint_ptr[id_a];
                }

                // Initialize gradient accumulators
                Tvec grad_d  = {0, 0, 0}; // Density gradient
                Tvec grad_u  = {0, 0, 0}; // Internal energy difference gradient
                Tvec grad_vx = {0, 0, 0}; // Velocity component gradients
                Tvec grad_vy = {0, 0, 0};
                Tvec grad_vz = {0, 0, 0};

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    Tvec dr    = xyz_a - xyz_acc[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);

                    if (rab2 > dint || id_a == id_b) {
                        return;
                    }

                    Tscal rab = sycl::sqrt(rab2);

                    // Kernel gradient: ∇W = (dW/dr) * (r/|r|)
                    Tscal dWdr = Kernel::dW_3d(rab, h_a);
                    Tvec gradW = dr * (dWdr * sham::inv_sat_positive(rab));

                    // Accumulate gradients (reference: g_pre_interaction.cpp lines 130-147)
                    grad_d += gradW * pmass;

                    // Internal energy gradient for pressure
                    Tscal u_b = (uint_ptr != nullptr) ? uint_ptr[id_b] : Tscal(0);
                    grad_u += gradW * (pmass * (u_b - u_a));

                    // Velocity gradients
                    Tvec v_b = v_acc[id_b];
                    grad_vx += gradW * (pmass * (v_b[0] - v_a[0]));
                    grad_vy += gradW * (pmass * (v_b[1] - v_a[1]));
                    grad_vz += gradW * (pmass * (v_b[2] - v_a[2]));
                });

                // Store density gradient
                grad_d_acc[id_a] = grad_d;

                // Compute pressure gradient: ∇P = (∇ρ * u + du) * (γ - 1)
                // (reference: g_pre_interaction.cpp line 143)
                Tvec grad_p      = (grad_d * u_a + grad_u) * (gamma - Tscal(1));
                grad_p_acc[id_a] = grad_p;

                // Normalize velocity gradients by density
                // (reference: g_pre_interaction.cpp lines 144-147)
                Tscal rho_inv     = sham::inv_sat_positive(rho_a);
                grad_vx_acc[id_a] = grad_vx * rho_inv;
                grad_vy_acc[id_a] = grad_vy * rho_inv;
                grad_vz_acc[id_a] = grad_vz * rho_inv;
            });
        });

        // Complete event states
        pcache.complete_event_state({e});
        buf_xyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        dens_field.get_buf().complete_event_state(e);
        grad_d_field.get_buf().complete_event_state(e);
        grad_p_field.get_buf().complete_event_state(e);
        grad_vx_buf.get_buf().complete_event_state(e);
        grad_vy_buf.get_buf().complete_event_state(e);
        grad_vz_buf.get_buf().complete_event_state(e);
        if (has_uint) {
            pdat.get_field_buf_ref<Tscal>(iuint).complete_event_state(e);
        }
    });
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

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

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

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                u32 cnt = pdat.get_obj_cnt();
                if (cnt == 0)
                    return;

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
typename shammodels::gsph::Solver<Tvec, Kern>::Tscal shammodels::gsph::Solver<Tvec, Kern>::
    compute_dt_cfl() {
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
        if (cnt == 0)
            return;

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
                if (!sycl::isfinite(h_i) || h_i <= Tscal(0))
                    h_i = Tscal(1e-10);
                if (!sycl::isfinite(cs_i) || cs_i <= Tscal(0))
                    cs_i = Tscal(1e-10);
                if (!sycl::isfinite(abs_a))
                    abs_a = Tscal(1e30);

                // Sound CFL condition: dt = C_cour * h / c_s
                // Following Kitajima et al. (2025) simple form for GSPH
                Tscal dt_c = C_cour * h_i / cs_i;

                // Force condition: dt = C_force * sqrt(h / |a|)
                Tscal dt_f = C_force * sycl::sqrt(h_i / (abs_a + Tscal(1e-30)));

                Tscal dt_min = sycl::min(dt_c, dt_f);

                // Ensure a valid finite timestep with minimum floor
                if (!sycl::isfinite(dt_min) || dt_min <= Tscal(0)) {
                    dt_min = Tscal(1e-10); // Minimum timestep floor
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
        rank_dt = Tscal(1e-6); // Reasonable floor for SPH simulations
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

    const u32 ivxyz = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz = pdl.get_field_idx<Tvec>("axyz");

    Tscal half_dt = Tscal{0.5} * dt;

    // Corrector: v = v + 0.5*a_new*dt (completing the leapfrog kick)
    // Predictor already added 0.5*a_old*dt, so total is 0.5*(a_old + a_new)*dt
    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &vxyz = pdat.get_field<Tvec>(ivxyz);
            auto &axyz = pdat.get_field<Tvec>(iaxyz);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{axyz.get_buf()},
                sham::MultiRef{vxyz.get_buf()},
                cnt,
                [half_dt](u32 i, const Tvec *axyz_new, Tvec *vxyz) {
                    vxyz[i] += half_dt * axyz_new[i];
                });
        });

    if (solver_config.has_field_uint()) {
        const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
        const u32 iduint = pdl.get_field_idx<Tscal>("duint");

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                u32 cnt = pdat.get_obj_cnt();
                if (cnt == 0)
                    return;

                auto &uint_field = pdat.get_field<Tscal>(iuint);
                auto &duint      = pdat.get_field<Tscal>(iduint);

                auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

                // Corrector: u = u + 0.5*du_new*dt (completing the leapfrog)
                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{duint.get_buf()},
                    sham::MultiRef{uint_field.get_buf()},
                    cnt,
                    [half_dt](u32 i, const Tscal *duint_new, Tscal *uint) {
                        uint[i] += half_dt * duint_new[i];
                    });
            });

        storage.old_duint.reset();
    }

    storage.old_axyz.reset();

    return true;
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::update_sync_load_values() {}

template<class Tvec, template<class> class Kern>
shammodels::gsph::TimestepLog shammodels::gsph::Solver<Tvec, Kern>::evolve_once() {

    // Validate configuration before running
    solver_config.check_config_runtime();

    Tscal t_current = solver_config.get_time();
    Tscal dt        = solver_config.get_dt();

    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        shamcomm::logs::raw_ln(
            shambase::format(
                "---------------- GSPH t = {}, dt = {} ----------------", t_current, dt));
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

    const bool is_sr = solver_config.is_sr_enabled();

    // STEP 1: PREDICTOR - move particles using OLD accelerations/derivatives
    // (On first iteration, accelerations are zero, so this is just position drift)
    if (is_sr) {
        // SR mode: use conserved variable integration
        // Initialize SR fields on first timestep (prim2cons)
        sr_init_fields();

        // SR Predictor: S += dS*dt/2; e += de*dt/2; cons2prim; x += v*dt
        // Skip on first timestep (dt=0) since nothing changes and density_field not yet computed
        if (dt > Tscal{0}) {
            sr_do_predictor(dt);
        }
    } else {
        // Newtonian mode: use leapfrog
        do_predictor_leapfrog(dt);
    }

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

    // STEP 4b: GRADIENTS - compute for MUSCL reconstruction (if enabled)
    // Computed BEFORE ghost communication so gradients are included in ghost data
    // Gradients are computed on local particles using neighbor data
    compute_gradients();

    // Initialize ghost layout BEFORE communication
    // (includes gradients if MUSCL is enabled)
    init_ghost_layout();

    // Communicate ghost fields (hpart, uint, vxyz, omega, and gradients if MUSCL)
    // This MUST happen BEFORE compute_eos_fields so EOS can be computed for ghosts
    communicate_merge_ghosts_fields();

    // STEP 4c: EOS - compute AFTER ghost communication (CRITICAL!)
    // This ensures P and cs are computed for ALL particles (local + ghost)
    // Following SPH pattern: EOS is computed on merged_patchdata_ghost
    compute_eos_fields();

    // SR-GSPH: Initialize conserved variables from primitives on first timestep
    // Must happen AFTER EOS is computed (need P, ρ) but BEFORE forces
    if (is_sr && !storage.sr_initialized) {
        sr_init_conserved();
    }

    // STEP 5: FORCES - compute accelerations/derivatives using FRESH EOS
    if (!is_sr) {
        // Newtonian mode: save old accelerations for corrector
        prepare_corrector();
    }

    // Update derivatives using GSPH Riemann solver
    // For SR mode, this stores dS/dt and de/dt to dedicated fields
    update_derivs();

    // STEP 6: CORRECTOR - refine velocities/conserved variables
    if (is_sr) {
        // SR mode: S += dS*dt/2; e += de*dt/2; cons2prim
        sr_apply_corrector(dt);
    } else {
        // Newtonian mode: v += a*dt/2
        apply_corrector(dt, Npart_all);
    }

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

// =========================================================================
// SR-GSPH integration methods
// =========================================================================

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::sr_init_fields() {
    StackEntry stack_loc{};

    using namespace shamrock::solvergraph;
    using namespace shamrock::patch;

    // Create sizes by iterating over non-empty patches (same pattern as compute_omega)
    std::shared_ptr<Indexes<u32>> sizes = std::make_shared<Indexes<u32>>("sr_sizes", "N");
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
    });

    // Initialize S_momentum field (nvar=1 for vector type)
    if (!storage.S_momentum) {
        storage.S_momentum = std::make_shared<Field<Tvec>>(1, "S_momentum", "S");
    }
    storage.S_momentum->ensure_sizes(sizes->indexes);

    // Initialize e_energy field (nvar=1 for scalar)
    if (!storage.e_energy) {
        storage.e_energy = std::make_shared<Field<Tscal>>(1, "e_energy", "e");
    }
    storage.e_energy->ensure_sizes(sizes->indexes);

    // Initialize dS_momentum field (derivatives)
    if (!storage.dS_momentum) {
        storage.dS_momentum = std::make_shared<Field<Tvec>>(1, "dS_momentum", "\\dot{S}");
    }
    storage.dS_momentum->ensure_sizes(sizes->indexes);

    // Initialize de_energy field (derivatives)
    if (!storage.de_energy) {
        storage.de_energy = std::make_shared<Field<Tscal>>(1, "de_energy", "\\dot{e}");
    }
    storage.de_energy->ensure_sizes(sizes->indexes);
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::sr_init_conserved() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    if (storage.sr_initialized) {
        return;
    }

    // Initialize SR fields if not done
    sr_init_fields();

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

    const Tscal c_speed   = solver_config.sr_config.get_c_speed();
    const Tscal gamma_eos = solver_config.get_eos_gamma();
    const Tscal pmass     = solver_config.gpart_mass;

    shamrock::solvergraph::Field<Tvec> &S_field  = shambase::get_check_ref(storage.S_momentum);
    shamrock::solvergraph::Field<Tscal> &e_field = shambase::get_check_ref(storage.e_energy);
    shamrock::solvergraph::Field<Tscal> &density_field  = shambase::get_check_ref(storage.density);
    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
        sham::DeviceBuffer<Tvec> &buf_S    = S_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_e   = e_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_rho = density_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_P   = pressure_field.get_buf(cur_p.id_patch);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{buf_vxyz, buf_rho, buf_P},
            sham::MultiRef{buf_S, buf_e},
            cnt,
            [c_speed, gamma_eos, pmass](
                u32 i, const Tvec *vxyz, const Tscal *rho_sph, const Tscal *P, Tvec *S, Tscal *e) {
                const Tscal c2 = c_speed * c_speed;

                // Lorentz factor
                const Tscal v2 = sycl::dot(vxyz[i], vxyz[i]) / c2;
                const Tscal gamma_lor
                    = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2, Tscal{1e-10}));

                // SPH density is lab-frame N = ν·ΣW (uses lab-frame positions)
                // Convert to rest-frame density: n = N / γ
                const Tscal N_lab = sycl::fmax(rho_sph[i], Tscal{1e-30});
                const Tscal n     = N_lab / gamma_lor;

                // Internal energy per unit mass (uses rest-frame density n)
                const Tscal u_int = P[i] / ((gamma_eos - Tscal{1}) * n);

                // Specific enthalpy: H = 1 + u/c² + P/(nc²)
                const Tscal H = Tscal{1} + u_int / c2 + P[i] / (n * c2);

                // Canonical momentum: S = γHv
                S[i] = vxyz[i] * (gamma_lor * H);

                // Canonical energy: e = γH - P/(Nc²) (uses lab-frame N)
                e[i] = gamma_lor * H - P[i] / (N_lab * c2);
            });
    });

    storage.sr_initialized = true;

    if (shamcomm::world_rank() == 0) {
        shamcomm::logs::raw_ln("SR-GSPH: Initialized conserved variables (S, e) from primitives");
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::sr_do_predictor(Tscal dt) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

    const Tscal half_dt   = dt / Tscal{2};
    const Tscal c_speed   = solver_config.sr_config.get_c_speed();
    const Tscal gamma_eos = solver_config.get_eos_gamma();

    shamrock::solvergraph::Field<Tvec> &S_field   = shambase::get_check_ref(storage.S_momentum);
    shamrock::solvergraph::Field<Tscal> &e_field  = shambase::get_check_ref(storage.e_energy);
    shamrock::solvergraph::Field<Tvec> &dS_field  = shambase::get_check_ref(storage.dS_momentum);
    shamrock::solvergraph::Field<Tscal> &de_field = shambase::get_check_ref(storage.de_energy);
    shamrock::solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);

    // Step 1: S += dS * dt/2, e += de * dt/2
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        sham::DeviceBuffer<Tvec> &buf_S   = S_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_e  = e_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{buf_dS, buf_de},
            sham::MultiRef{buf_S, buf_e},
            cnt,
            [half_dt](u32 i, const Tvec *dS, const Tscal *de, Tvec *S, Tscal *e) {
                S[i] += dS[i] * half_dt;
                e[i] += de[i] * half_dt;
            });
    });

    // Step 2: Primitive recovery (cons2prim) to get v
    sr_cons2prim();

    // Step 3: Position drift: x += v * dt
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &xyz_field  = pdat.get_field<Tvec>(ixyz);
        auto &vxyz_field = pdat.get_field<Tvec>(ivxyz);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{vxyz_field.get_buf()},
            sham::MultiRef{xyz_field.get_buf()},
            cnt,
            [dt](u32 i, const Tvec *vxyz, Tvec *xyz) {
                xyz[i] += vxyz[i] * dt;
            });
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::sr_apply_corrector(Tscal dt) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    const Tscal half_dt = dt / Tscal{2};

    shamrock::solvergraph::Field<Tvec> &S_field   = shambase::get_check_ref(storage.S_momentum);
    shamrock::solvergraph::Field<Tscal> &e_field  = shambase::get_check_ref(storage.e_energy);
    shamrock::solvergraph::Field<Tvec> &dS_field  = shambase::get_check_ref(storage.dS_momentum);
    shamrock::solvergraph::Field<Tscal> &de_field = shambase::get_check_ref(storage.de_energy);

    // S += dS * dt/2, e += de * dt/2 (second half-step)
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        sham::DeviceBuffer<Tvec> &buf_S   = S_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_e  = e_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{buf_dS, buf_de},
            sham::MultiRef{buf_S, buf_e},
            cnt,
            [half_dt](u32 i, const Tvec *dS, const Tscal *de, Tvec *S, Tscal *e) {
                S[i] += dS[i] * half_dt;
                e[i] += de[i] * half_dt;
            });
    });

    // Primitive recovery to update v, P, ρ for next step
    sr_cons2prim();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::sr_cons2prim() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const bool has_uint       = solver_config.has_field_uint();
    const u32 iuint           = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;

    const Tscal c_speed   = solver_config.sr_config.get_c_speed();
    const Tscal gamma_eos = solver_config.get_eos_gamma();
    const Tscal pmass     = solver_config.gpart_mass;

    shamrock::solvergraph::Field<Tvec> &S_field  = shambase::get_check_ref(storage.S_momentum);
    shamrock::solvergraph::Field<Tscal> &e_field = shambase::get_check_ref(storage.e_energy);
    shamrock::solvergraph::Field<Tscal> &density_field  = shambase::get_check_ref(storage.density);
    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
        sham::DeviceBuffer<Tvec> &buf_S    = S_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_e   = e_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_rho = density_field.get_buf(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_P   = pressure_field.get_buf(cur_p.id_patch);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        // Also update uint field if available (for post-processing access)
        if (has_uint) {
            sham::DeviceBuffer<Tscal> &buf_uint = pdat.get_field_buf_ref<Tscal>(iuint);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_S, buf_e, buf_rho},
                sham::MultiRef{buf_vxyz, buf_P, buf_uint},
                cnt,
                [c_speed, gamma_eos](
                    u32 i,
                    const Tvec *S,
                    const Tscal *e,
                    const Tscal *rho_sph,
                    Tvec *vxyz,
                    Tscal *P,
                    Tscal *uint_out) {
                    // Get momentum magnitude and direction
                    const Tscal S_mag = sycl::sqrt(sycl::dot(S[i], S[i]));
                    const Tvec S_dir  = (S_mag > Tscal{1e-15}) ? S[i] / S_mag : Tvec{1, 0, 0};

                    // SPH density summation gives lab-frame density N = ν·ΣW
                    const Tscal N = sycl::fmax(rho_sph[i], Tscal{1e-30});

                    // Primitive recovery takes lab-frame N and converts to n = N/γ internally
                    sr::SRPrimitiveVars<Tscal> prim = sr::conserved_to_primitive<Tscal>(
                        S_mag, Tscal{0}, e[i], N, gamma_eos, c_speed);

                    // Reconstruct velocity vector
                    vxyz[i] = S_dir * prim.vel_normal * c_speed;

                    // Update pressure
                    P[i] = sycl::fmax(prim.pressure, Tscal{1e-10});

                    // Update uint for post-processing compatibility with test scripts
                    // Test computes: P = (γ-1) * ρ_sph * u, where ρ_sph is lab-frame N
                    // So we need: u = P / ((γ-1) * N) to ensure P = (γ-1) * N * u = P
                    uint_out[i] = P[i] / ((gamma_eos - Tscal{1}) * N);
                });
        } else {
            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_S, buf_e, buf_rho},
                sham::MultiRef{buf_vxyz, buf_P},
                cnt,
                [c_speed, gamma_eos](
                    u32 i,
                    const Tvec *S,
                    const Tscal *e,
                    const Tscal *rho_sph,
                    Tvec *vxyz,
                    Tscal *P) {
                    // Get momentum magnitude and direction
                    const Tscal S_mag = sycl::sqrt(sycl::dot(S[i], S[i]));
                    const Tvec S_dir  = (S_mag > Tscal{1e-15}) ? S[i] / S_mag : Tvec{1, 0, 0};

                    // SPH density summation gives lab-frame density N = ν·ΣW
                    const Tscal N = sycl::fmax(rho_sph[i], Tscal{1e-30});

                    // Primitive recovery takes lab-frame N and converts to n = N/γ internally
                    sr::SRPrimitiveVars<Tscal> prim = sr::conserved_to_primitive<Tscal>(
                        S_mag, Tscal{0}, e[i], N, gamma_eos, c_speed);

                    // Reconstruct velocity vector
                    vxyz[i] = S_dir * prim.vel_normal * c_speed;

                    // Update pressure
                    P[i] = sycl::fmax(prim.pressure, Tscal{1e-10});
                });
        }
    });
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
