// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
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

#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/gather_str.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/Solver.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/config/FieldNames.hpp"
#include "shammodels/gsph/modules/ComputeLoadBalanceValue.hpp"
#include "shammodels/gsph/modules/GSPHUtilities.hpp"
#include "shammodels/gsph/modules/UpdateDerivs.hpp"
#include "shammodels/gsph/modules/io/VTKDump.hpp"
#include "shammodels/sph/modules/ComputeOmega.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensityNeighLim.hpp"
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

    storage.part_counts = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        edges::part_counts, "N_{\\rm part}");

    storage.part_counts_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        edges::part_counts_with_ghost, "N_{\\rm part, with ghost}");

    storage.patch_rank_owner = std::make_shared<shamrock::solvergraph::RankGetter>(
        [&](u64 patch_id) -> u32 {
            return scheduler().get_patch_rank_owner(patch_id);
        },
        "patch_rank_owner",
        "rank");

    // Merged ghost spans
    storage.positions_with_ghosts = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>(
        edges::positions_with_ghosts, "\\mathbf{r}");
    storage.hpart_with_ghosts
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>(edges::hpart_with_ghosts, "h");

    storage.neigh_cache
        = std::make_shared<shammodels::sph::solvergraph::NeighCache>(edges::neigh_cache, "neigh");

    // Register ghost handler in solvergraph for explicit data dependency tracking
    storage.ghost_handler = storage.solver_graph.register_edge(
        "ghost_handler", solvergraph::GhostHandlerEdge<Tvec>("ghost_handler", "\\mathcal{G}"));

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

    using CfgClass = gsph::GSPHGhostHandlerConfig<Tvec>;
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
        shambase::get_check_ref(storage.ghost_handler)
            .set(
                GhostHandle{
                    scheduler(), BCFree{}, storage.patch_rank_owner, storage.xyzh_ghost_layout});
    } else if (
        SolverBCPeriodic *c
        = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
        shambase::get_check_ref(storage.ghost_handler)
            .set(
                GhostHandle{
                    scheduler(),
                    BCPeriodic{},
                    storage.patch_rank_owner,
                    storage.xyzh_ghost_layout});
    } else if (
        SolverBCShearingPeriodic *c
        = std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)) {
        // Shearing periodic boundaries (Stone 2010) - reuse SPH implementation
        shambase::get_check_ref(storage.ghost_handler)
            .set(
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

    using GSPHUtils = GSPHUtilities<Tvec, Kernel>;
    GSPHUtils gsph_utils(scheduler());

    // Same widening as compute_presteps_rint()/start_neighbors_cache(): with
    // InutsukaV2 the kernel support is sqrt(2)*h, so the patch/rank ghost
    // interface radius must widen too, or particles near a patch boundary
    // could be missing valid neighbors from the adjacent patch.
    Tscal h_evol_max = solver_config.htol_up_coarse_cycle;
    if (solver_config.is_force_inutsuka_v2()) {
        h_evol_max *= shambase::constants::sqrt_2<Tscal>;
    }

    storage.ghost_patch_cache.set(gsph_utils.build_interf_cache(
        shambase::get_check_ref(storage.ghost_handler).get(),
        storage.serial_patch_tree.get(),
        h_evol_max));
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
        shambase::get_check_ref(storage.ghost_handler)
            .get()
            .build_comm_merge_positions(storage.ghost_patch_cache.get()));

    // Get field indices from xyzh_ghost_layout
    const u32 ixyz_ghost
        = storage.xyzh_ghost_layout->template get_field_idx<Tvec>(gsph::names::common::xyz);
    const u32 ihpart_ghost
        = storage.xyzh_ghost_layout->template get_field_idx<Tscal>(gsph::names::common::hpart);

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
                [&, ixyz_ghost](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tvec>(ixyz_ghost));
                }));

    shambase::get_check_ref(storage.hpart_with_ghosts)
        .set_refs(
            storage.merged_xyzh.get().template map<std::reference_wrapper<PatchDataField<Tscal>>>(
                [&, ihpart_ghost](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tscal>(ihpart_ghost));
                }));
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::build_merged_pos_trees() {
    StackEntry stack_loc{};

    auto &merged_xyzh = storage.merged_xyzh.get();
    auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

    // Get field index from xyzh_ghost_layout
    const u32 ixyz_ghost
        = storage.xyzh_ghost_layout->template get_field_idx<Tvec>(gsph::names::common::xyz);

    shambase::DistributedData<RTree> trees = merged_xyzh.template map<RTree>(
        [&, ixyz_ghost](u64 id, shamrock::patch::PatchDataLayer &merged) {
            PatchDataField<Tvec> &pos = merged.template get_field<Tvec>(ixyz_ghost);
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

    // The Inutsuka V2 force formulation evaluates the kernel gradient at an
    // effective smoothing length sqrt(2)*h (Inutsuka 2002). This tree-node
    // interaction-range field is used to prune tree traversal in
    // start_neighbors_cache(), so it must be widened by the same factor, or
    // whole subtrees containing valid sqrt(2)*h-range neighbors get pruned
    // before the leaf-level search even runs.
    Tscal htol = solver_config.htol_up_coarse_cycle;
    if (solver_config.is_force_inutsuka_v2()) {
        htol *= shambase::constants::sqrt_2<Tscal>;
    }

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
                    [htol](u32 i, Tscal *h_tree) {
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

    // The Inutsuka V2 force formulation evaluates the kernel gradient at an
    // effective smoothing length sqrt(2)*h (Inutsuka 2002), so its support radius
    // is sqrt(2) times larger than the standard h*Rkern cutoff used below. Widen
    // the cached search radius accordingly, or pairs in (h*Rkern, sqrt(2)*h*Rkern)
    // would silently be missing from the cache for that formulation.
    if (solver_config.is_force_inutsuka_v2()) {
        h_tolerance *= shambase::constants::sqrt_2<Tscal>;
    }

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

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_neighbors_cache() {
    storage.neigh_cache->neigh_cache = {};
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::compute_density() {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    auto dev_sched               = shamsys::instance::get_compute_scheduler_ptr();
    const Tscal pmass            = solver_config.gpart_mass;
    static constexpr Tscal Rkern = Kernel::Rkern;

    if (pmass == 0) {
        shambase::throw_with_loc<std::runtime_error>(
            "invalid gpart_mass 0 in compute_density, this configuration can not converge.");
    }

    shamrock::solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);

    // Density is only needed for local particles here; it gets propagated to
    // ghosts afterwards by communicate_merge_ghosts_fields().
    shambase::DistributedData<u32> &counts = shambase::get_check_ref(storage.part_counts).indexes;
    density_field.ensure_sizes(counts);

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ihpart          = pdl.get_field_idx<Tscal>(gsph::names::common::hpart);

    auto &merged_xyzh = storage.merged_xyzh.get();
    auto &neigh_cache = storage.neigh_cache->neigh_cache;

    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0)
            return;

        auto &mfield = merged_xyzh.get(p.id_patch);
        auto &pcache = neigh_cache.get(p.id_patch);

        // Position from merged data (includes ghosts for neighbor search)
        auto &buf_xyz   = mfield.template get_field_buf_ref<Tvec>(0);
        auto &buf_hpart = pdat.get_field_buf_ref<Tscal>(ihpart);

        auto &dens_field = density_field.get_field(p.id_patch);

        sham::DeviceQueue &q = dev_sched->get_queue();
        sham::EventList depends_list;

        auto ploop_ptrs  = pcache.get_read_access(depends_list);
        auto xyz_acc     = buf_xyz.get_read_access(depends_list);
        auto h_acc       = buf_hpart.get_read_access(depends_list);
        auto density_acc = dens_field.get_buf().get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, pmass](sycl::handler &cgh) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            shambase::parallel_for(cgh, cnt, "gsph_compute_density", [=](u64 gid) {
                u32 id_a = (u32) gid;

                Tvec xyz_a = xyz_acc[id_a];
                Tscal h_a  = h_acc[id_a];
                Tscal dint = h_a * h_a * Rkern * Rkern;

                // SPH density summation: rho_a = sum_b m_b W(|r_a - r_b|, h_a)
                // (includes the self contribution b == a)
                Tscal rho_sum = pmass * Kernel::W_3d(Tscal(0), h_a);

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    if (id_b == id_a) {
                        return;
                    }

                    Tvec dr    = xyz_a - xyz_acc[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);

                    if (rab2 > dint) {
                        return;
                    }

                    Tscal rab = sycl::sqrt(rab2);
                    rho_sum += pmass * Kernel::W_3d(rab, h_a);
                });

                density_acc[id_a] = sycl::max(rho_sum, Tscal(1e-30));
            });
        });

        pcache.complete_event_state({e});
        buf_xyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        dens_field.get_buf().complete_event_state(e);
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::gsph_prestep(Tscal time_val, Tscal dt) {
    StackEntry stack_loc{};

    shamlog_debug_ln("GSPH", "Prestep at t =", time_val, "dt =", dt);

    using namespace shamrock;
    using namespace shamrock::patch;

    using RTree    = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;
    using SPHUtils = gsph::GSPHUtilities<Tvec, Kernel>;

    SPHUtils sph_utils(scheduler());
    shamrock::SchedulerUtility utility(scheduler());

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");

    ComputeField<Tscal> _epsilon_h, _h_old;

    auto should_set_omega_mask = std::make_shared<shamrock::solvergraph::Field<u32>>(
        1, "should_set_omega_mask", "should_set_omega_mask");

    u32 hstep_cnt = 0;
    u32 hstep_max = solver_config.h_max_subcycles_count;
    for (; hstep_cnt < hstep_max; hstep_cnt++) {

        gen_ghost_handler(time_val + dt);
        build_ghost_cache();
        merge_position_ghost();
        build_merged_pos_trees();
        compute_presteps_rint();
        start_neighbors_cache();

        _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1, Tscal(100));
        _h_old     = utility.save_field<Tscal>(ihpart, "h_old");

        Tscal max_eps_h;

        if (solver_config.gpart_mass == 0) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "invalid gpart_mass {}, this configuration can not converge.\n"
                "Please set it using either model.set_particle_mass(pmass) or "
                "cfg.set_particle_mass(pmass)",
                solver_config.gpart_mass));
        }

        // sizes
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes
            = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        // neigh cache
        auto &neigh_cache = storage.neigh_cache;

        // positions
        auto &pos_merged = storage.positions_with_ghosts;

        // old smoothing length field
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hold
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hold_refs = {};
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &field = _h_old.get_field(p.id_patch);
            hold_refs.add_obj(p.id_patch, std::ref(field));
        });
        hold->set_refs(hold_refs);

        // new smoothing length field
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hnew
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &field = pdat.get_field<Tscal>(ihpart);
            hnew_refs.add_obj(p.id_patch, std::ref(field));
        });
        hnew->set_refs(hnew_refs);

        // epsilon field
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> eps_h
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> eps_h_refs = {};
        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &field = _epsilon_h.get_field(p.id_patch);
            eps_h_refs.add_obj(p.id_patch, std::ref(field));
        });
        eps_h->set_refs(eps_h_refs);

        std::shared_ptr<shamrock::solvergraph::INode> smth_h_iter_ptr;

        using h_conf_density_based = typename gsph::SmoothingLengthConfig::DensityBased;
        using h_conf_neigh_lim     = typename gsph::SmoothingLengthConfig::DensityBasedNeighLim;

        if (h_conf_density_based *conf
            = std::get_if<h_conf_density_based>(&solver_config.smoothing_length_config.config)) {
            std::shared_ptr<shammodels::sph::modules::IterateSmoothingLengthDensity<Tvec, Kernel>>
                smth_h_iter = std::make_shared<
                    shammodels::sph::modules::IterateSmoothingLengthDensity<Tvec, Kernel>>(
                    solver_config.gpart_mass,
                    solver_config.htol_up_coarse_cycle,
                    solver_config.htol_up_fine_cycle);
            smth_h_iter->set_edges(sizes, neigh_cache, pos_merged, hold, hnew, eps_h);
            smth_h_iter_ptr = smth_h_iter;
        } else if (
            h_conf_neigh_lim *conf
            = std::get_if<h_conf_neigh_lim>(&solver_config.smoothing_length_config.config)) {
            std::shared_ptr<
                shammodels::sph::modules::IterateSmoothingLengthDensityNeighLim<Tvec, Kernel>>
                smth_h_iter_neigh_lim = std::make_shared<
                    shammodels::sph::modules::IterateSmoothingLengthDensityNeighLim<Tvec, Kernel>>(
                    solver_config.gpart_mass,
                    solver_config.htol_up_coarse_cycle,
                    solver_config.htol_up_fine_cycle,
                    conf->max_neigh_count);
            smth_h_iter_neigh_lim->set_edges(
                sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
            smth_h_iter_ptr = smth_h_iter_neigh_lim;
        } else {
            shambase::throw_with_loc<std::runtime_error>("Invalid smoothing length configuration");
        }
        // iterate smoothing length

        std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> is_converged
            = std::make_shared<shamrock::solvergraph::ScalarEdge<bool>>("", "");

        shammodels::sph::modules::LoopSmoothingLengthIter<Tvec> loop_smth_h_iter(
            smth_h_iter_ptr, solver_config.epsilon_h, solver_config.h_iter_per_subcycles, false);
        loop_smth_h_iter.set_edges(eps_h, is_converged);

        loop_smth_h_iter.evaluate();

        if (!is_converged->value) {

            Tscal largest_h = 0;

            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                largest_h = sham::max(largest_h, pdat.get_field<Tscal>(ihpart).compute_max());
            });
            Tscal global_largest_h = shamalgs::collective::allreduce_max(largest_h);

            std::string add_info = "";
            u64 cnt_unconverged  = 0;
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto res
                    = _epsilon_h.get_field(p.id_patch).get_ids_buf_where([](auto access, u32 id) {
                          return access[id] == -1;
                      });

                if (hstep_cnt == hstep_max - 1) {
                    if (std::get<0>(res)) {
                        add_info += "\n    patch " + std::to_string(p.id_patch) + " ";
                        add_info += "errored parts : \n";
                        sycl::buffer<u32> &idx_err = *std::get<0>(res);

                        sham::DeviceBuffer<Tvec> &xyz    = pdat.get_field_buf_ref<Tvec>(0);
                        sham::DeviceBuffer<Tscal> &hpart = pdat.get_field_buf_ref<Tscal>(ihpart);

                        auto pos = xyz.copy_to_stdvec();
                        auto h   = hpart.copy_to_stdvec();

                        {
                            sycl::host_accessor acc{idx_err};
                            for (u32 i = 0; i < idx_err.size(); i++) {
                                add_info += sham::format(
                                    "{} - pos : {}, hpart : {}\n", acc[i], pos[acc[i]], h[acc[i]]);
                            }
                        }
                    }
                }

                cnt_unconverged += std::get<1>(res);
            });

            u64 global_cnt_unconverged = shamalgs::collective::allreduce_sum(cnt_unconverged);

            if (shamcomm::world_rank() == 0) {
                logger::warn_ln(
                    "Smoothinglength",
                    "smoothing length is not converged, rerunning the iterator ...\n     largest h "
                    "=",
                    global_largest_h,
                    "unconverged cnt =",
                    global_cnt_unconverged,
                    add_info);
            }

            reset_ghost_handler();
            clear_ghost_cache();

            shambase::get_check_ref(storage.part_counts).free_alloc();
            shambase::get_check_ref(storage.part_counts_with_ghost).free_alloc();
            shambase::get_check_ref(storage.positions_with_ghosts).free_alloc();
            shambase::get_check_ref(storage.hpart_with_ghosts).free_alloc();

            storage.merged_xyzh.reset();

            clear_merged_pos_trees();
            reset_presteps_rint();
            reset_neighbors_cache();

            // scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            //     pdat.synchronize_buf();
            // });

            continue;
        }

        // The hpart is not valid anymore in ghost zones since we iterated it's value
        shambase::get_check_ref(storage.hpart_with_ghosts).free_alloc();

        _epsilon_h.reset();
        _h_old.reset();
        break;
    }

    if (hstep_cnt == hstep_max) {
        logger::err_ln("GSPH", "the h iterator is not converged after", hstep_cnt, "iterations");
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hnew_edge
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
    shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        auto &field = pdat.get_field<Tscal>(ihpart);
        hnew_refs.add_obj(p.id_patch, std::ref(field));
    });
    hnew_edge->set_refs(hnew_refs);

    sph::modules::NodeComputeOmega<Tvec, Kern> compute_omega{solver_config.gpart_mass};
    compute_omega.set_edges(
        storage.part_counts,
        storage.neigh_cache,
        storage.positions_with_ghosts,
        hnew_edge,
        storage.omega);
    compute_omega.evaluate();

    if (solver_config.smoothing_length_config.is_density_based_neigh_lim()) {
        // if the h limiter is triggered, omega does not hold it's sense of dh/dr anymore
        // so we set it to 1, this effectively is equivalent of disabling the energy correction
        // term corresponding to dh/dr
        sph::modules::SetWhenMask<Tscal> set_omega_mask{1};
        set_omega_mask.set_edges(storage.part_counts, should_set_omega_mask, storage.omega);
        set_omega_mask.evaluate();
    }

    // NodeComputeOmega only produces the grad-h correction factor, not density
    // itself (see Solver.hpp::compute_density() for why GSPH needs an explicit
    // summed density field, unlike plain SPH). Compute it now, on the same
    // converged h / neighbor cache / merged positions used above.
    compute_density();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::apply_position_boundary(Tscal time_val) {
    StackEntry stack_loc{};

    shamlog_debug_ln("GSPH", "apply position boundary");

    PatchScheduler &sched = scheduler();
    shamrock::SchedulerUtility integrators(sched);
    shamrock::ReattributeDataUtility reatrib(sched);

    auto &pdl         = sched.pdl_old();
    const u32 ixyz    = pdl.get_field_idx<Tvec>(gsph::names::common::xyz);
    const u32 ivxyz   = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
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

    reatrib.reatribute_patch_objects(storage.serial_patch_tree.get(), gsph::names::common::xyz);
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::do_predictor_leapfrog(Tscal dt) {
    StackEntry stack_loc{};
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ixyz            = pdl.get_field_idx<Tvec>(gsph::names::common::xyz);
    const u32 ivxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
    const u32 iaxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::axyz);

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;
    const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>(gsph::names::newtonian::duint) : 0;

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
    storage.xyzh_ghost_layout->template add_field<Tvec>(gsph::names::common::xyz, 1);
    storage.xyzh_ghost_layout->template add_field<Tscal>(gsph::names::common::hpart, 1);

    // Reset first in case it was set from a previous timestep
    storage.ghost_layout = std::make_shared<shamrock::patch::PatchDataLayerLayout>();

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

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ixyz            = pdl.get_field_idx<Tvec>(gsph::names::common::xyz);
    const u32 ivxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
    const u32 ihpart          = pdl.get_field_idx<Tscal>(gsph::names::common::hpart);

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;

    auto &ghost_layout_ptr                              = storage.ghost_layout;
    shamrock::patch::PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(ghost_layout_ptr);
    u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>(gsph::names::common::hpart);
    u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
    u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>(gsph::names::newtonian::omega);
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>(gsph::names::newtonian::density);
    u32 iuint_interf
        = has_uint ? ghost_layout.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;

    // Gradient field indices (for MUSCL reconstruction)
    const bool has_grads = solver_config.requires_gradients();
    u32 igrad_d_interf
        = has_grads ? ghost_layout.get_field_idx<Tvec>(gsph::names::newtonian::grad_density) : 0;
    u32 igrad_p_interf
        = has_grads ? ghost_layout.get_field_idx<Tvec>(gsph::names::newtonian::grad_pressure) : 0;
    u32 igrad_vx_interf
        = has_grads ? ghost_layout.get_field_idx<Tvec>(gsph::names::newtonian::grad_vx) : 0;
    u32 igrad_vy_interf
        = has_grads ? ghost_layout.get_field_idx<Tvec>(gsph::names::newtonian::grad_vy) : 0;
    u32 igrad_vz_interf
        = has_grads ? ghost_layout.get_field_idx<Tvec>(gsph::names::newtonian::grad_vz) : 0;

    using InterfaceBuildInfos = typename gsph::GSPHGhostHandler<Tvec>::InterfaceBuildInfos;

    gsph::GSPHGhostHandler<Tvec> &ghost_handle
        = shambase::get_check_ref(storage.ghost_handler).get();
    shamrock::solvergraph::Field<Tscal> &omega   = shambase::get_check_ref(storage.omega);
    shamrock::solvergraph::Field<Tscal> &density = shambase::get_check_ref(storage.density);

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

    timer_interf.stop();
    storage.timings_details.interface += timer_interf.elapsed_sec();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_merge_ghosts_fields() {
    storage.merged_patchdata_ghost.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::compute_eos_fields() {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    // GSPH EOS: Following reference implementation (g_pre_interaction.cpp)
    // P = (\gamma - 1) * \rho * u  where \rho is from SPH summation
    // c = sqrt(\gamma * (\gamma - 1) * u)  -- from internal energy, not from P/\rho

    auto dev_sched      = shamsys::instance::get_compute_scheduler_ptr();
    const Tscal gamma   = solver_config.get_eos_gamma();
    const bool has_uint = solver_config.has_field_uint();

    // Get ghost layout field indices
    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>(gsph::names::newtonian::density);
    u32 iuint_interf
        = has_uint ? ghost_layout.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;

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
                    // P = (\gamma - 1) * \rho * u
                    Tscal u = uint_ptr[i];
                    u       = sycl::max(u, Tscal(1e-30));
                    Tscal P = (gamma - Tscal(1.0)) * rho * u;

                    // Sound speed from internal energy (reference: solver.cpp line 2661)
                    // c = sqrt(\gamma * (\gamma - 1) * u)
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
void shammodels::gsph::Solver<Tvec, Kern>::copy_eos_to_patchdata() {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    // Copy density, pressure, and soundspeed from solvergraph fields to patchdata
    // This ensures the values persist across restarts and can be read by VTKDump

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    u32 idensity              = pdl.get_field_idx<Tscal>(names::newtonian::density);
    u32 ipressure             = pdl.get_field_idx<Tscal>(names::newtonian::pressure);
    u32 isoundspeed           = pdl.get_field_idx<Tscal>(names::newtonian::soundspeed);

    auto &density_field    = shambase::get_check_ref(storage.density);
    auto &pressure_field   = shambase::get_check_ref(storage.pressure);
    auto &soundspeed_field = shambase::get_check_ref(storage.soundspeed);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u32 npart = pdat.get_obj_cnt();
        if (npart == 0) {
            return;
        }

        // Get patchdata buffers
        sham::DeviceBuffer<Tscal> &buf_rho = pdat.get_field_buf_ref<Tscal>(idensity);
        sham::DeviceBuffer<Tscal> &buf_P   = pdat.get_field_buf_ref<Tscal>(ipressure);
        sham::DeviceBuffer<Tscal> &buf_cs  = pdat.get_field_buf_ref<Tscal>(isoundspeed);

        // Get solvergraph field buffers (source data)
        sham::DeviceBuffer<Tscal> &buf_rho_in = density_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_P_in   = pressure_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs_in = soundspeed_field.get_field(cur_p.id_patch).get_buf();

        auto &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto rho_in = buf_rho_in.get_read_access(depends_list);
        auto P_in   = buf_P_in.get_read_access(depends_list);
        auto cs_in  = buf_cs_in.get_read_access(depends_list);
        auto rho    = buf_rho.get_write_access(depends_list);
        auto P      = buf_P.get_write_access(depends_list);
        auto cs     = buf_cs.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>{npart}, [=](sycl::item<1> item) {
                rho[item] = rho_in[item];
                P[item]   = P_in[item];
                cs[item]  = cs_in[item];
            });
        });

        buf_rho_in.complete_event_state(e);
        buf_P_in.complete_event_state(e);
        buf_cs_in.complete_event_state(e);
        buf_rho.complete_event_state(e);
        buf_P.complete_event_state(e);
        buf_cs.complete_event_state(e);
    });
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

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ihpart          = pdl.get_field_idx<Tscal>(gsph::names::common::hpart);
    const u32 ivxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
    const bool has_uint       = solver_config.has_field_uint();
    const u32 iuint = has_uint ? pdl.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;

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
    // grad_d = \sigma_j m_j \nabla W_ij
    // grad_p = (grad_d * u_i + du) * (\gamma - 1)  where du = \sigma_j m_j (u_j - u_i) \nabla W_ij
    // grad_v = \sigma_j m_j (v_j - v_i) \nabla W_ij / \rho_i
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

                    // Kernel gradient: \nabla W = (dW/dr) * (r/|r|)
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

                // Compute pressure gradient: \nabla P = (\nabla \rho * u + du) * (\gamma - 1)
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
    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 iaxyz = pdl.get_field_idx<Tvec>(gsph::names::newtonian::axyz);

    // Create compute field to store old acceleration
    auto old_axyz = utility.make_compute_field<Tvec>(gsph::names::internal::old_axyz, 1);

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
        const u32 iduint = pdl.get_field_idx<Tscal>(gsph::names::newtonian::duint);
        auto old_duint   = utility.make_compute_field<Tscal>(gsph::names::internal::old_duint, 1);

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

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ihpart          = pdl.get_field_idx<Tscal>(gsph::names::common::hpart);
    const u32 iaxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::axyz);

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

    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 ivxyz = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
    const u32 iaxyz = pdl.get_field_idx<Tvec>(gsph::names::newtonian::axyz);

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
        const u32 iuint  = pdl.get_field_idx<Tscal>(gsph::names::newtonian::uint);
        const u32 iduint = pdl.get_field_idx<Tscal>(gsph::names::newtonian::duint);

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
void shammodels::gsph::Solver<Tvec, Kern>::update_sync_load_values() {
    modules::ComputeLoadBalanceValue<Tvec, Kern>(context, solver_config, storage)
        .update_load_balancing();
    scheduler().scheduler_step(false, false);
}

template<class Tvec, template<class> class Kern>
shammodels::gsph::TimestepLog shammodels::gsph::Solver<Tvec, Kern>::evolve_once() {

    // Validate configuration before running
    solver_config.check_config_runtime();

    Tscal t_current = get_time();
    Tscal dt        = get_dt();

    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        shamcomm::logs::raw_ln(
            sham::format("---------------- GSPH t = {}, dt = {} ----------------", t_current, dt));
    }

    shambase::Timer tstep;
    tstep.start();

    // Load balancing step
    gsph::modules::ComputeLoadBalanceValue<Tvec, Kern>(context, solver_config, storage)
        .update_load_balancing();
    scheduler().scheduler_step(true, true);
    gsph::modules::ComputeLoadBalanceValue<Tvec, Kern>(context, solver_config, storage)
        .update_load_balancing();
    scheduler().scheduler_step(false, false);

    /// patch_rank_owner is automatically updated since it is just a lambda

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
    // 3. PRESTEP: converge smoothing length h, then compute omega and density
    //    (density is now computed inside gsph_prestep(), on the same
    //    converged h / merged positions / neighbor cache - see compute_density())
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

    // STEP 3: PRESTEP - converges h, builds ghost/tree/neighbor-cache state on
    // the new positions, and computes omega + density (see gsph_prestep() /
    // compute_density()). Everything it builds (ghost handler, ghost cache,
    // merged positions, merged trees, rint field, neighbor cache) stays valid
    // and is reused below - no need to rebuild it a second time here.
    gsph_prestep(t_current, dt);

    // STEP 4a: GRADIENTS - compute for MUSCL reconstruction (if enabled)
    // Computed BEFORE ghost communication so gradients are included in ghost data
    // Gradients are computed on local particles using neighbor data
    compute_gradients();

    // Initialize ghost layout BEFORE communication
    // (includes gradients if MUSCL is enabled)
    init_ghost_layout();

    // Communicate ghost fields (hpart, uint, vxyz, omega, and gradients if MUSCL)
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

    // Copy EOS fields to patchdata for persistence and VTKDump access
    copy_eos_to_patchdata();

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
    set_time(t_current + dt);
    set_next_dt(dt_next);

    solve_logs.step_count++;

    tstep.stop();

    // Prepare timing log
    TimestepLog log;
    log.rank     = shamcomm::world_rank();
    log.rate     = Tscal(Npart_all) / tstep.elapsed_sec();
    log.npart    = Npart_all;
    log.tcompute = tstep.elapsed_sec();

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
