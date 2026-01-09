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
#include "shammodels/gsph/FieldNames.hpp"
#include "shammodels/gsph/Solver.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/core/PhysicsModeFactory.hpp"
#include "shammodels/gsph/modules/BoundaryHandler.hpp"
#include "shammodels/gsph/modules/BuildTrees.hpp"
#include "shammodels/gsph/modules/ComputeCFL.hpp"
#include "shammodels/gsph/modules/ComputeGradients.hpp"
#include "shammodels/gsph/modules/FunctorNode.hpp"
#include "shammodels/gsph/modules/GhostCommunicator.hpp"
#include "shammodels/gsph/modules/NeighbourCache.hpp"
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
#include "shamrock/solvergraph/IDataEdge.hpp"
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
    using namespace shamrock::solvergraph;
    using namespace shammodels::gsph::fields;
    using namespace shammodels::gsph::computed_fields;

    SolverGraph &solver_graph = storage.solver_graph;

    // Register index edges
    storage.part_counts
        = solver_graph.register_edge("part_counts", Indexes<u32>("part_counts", "N_{\\rm part}"));
    storage.part_counts_with_ghost = solver_graph.register_edge(
        "part_counts_with_ghost",
        Indexes<u32>("part_counts_with_ghost", "N_{\\rm part, with ghost}"));
    storage.patch_rank_owner = solver_graph.register_edge(
        "patch_rank_owner", ScalarsEdge<u32>("patch_rank_owner", "rank"));

    // Register field reference edges (merged ghost spans)
    storage.positions_with_ghosts
        = solver_graph.register_edge("part_pos", FieldRefs<Tvec>("part_pos", "\\mathbf{r}"));
    storage.hpart_with_ghosts
        = solver_graph.register_edge("h_part", FieldRefs<Tscal>("h_part", "h"));

    // Register neighbor cache
    storage.neigh_cache = solver_graph.register_edge(
        "neigh_cache", shammodels::sph::solvergraph::NeighCache("neigh_cache", "neigh"));

    // Register computed field edges (physics-agnostic only)
    // Note: density field is registered by physics mode via init_fields()
    storage.omega
        = solver_graph.register_edge(fields::OMEGA, Field<Tscal>(1, fields::OMEGA, "\\Omega"));
    storage.pressure = solver_graph.register_edge(
        computed_fields::PRESSURE, Field<Tscal>(1, computed_fields::PRESSURE, "P"));
    storage.soundspeed = solver_graph.register_edge(
        computed_fields::SOUNDSPEED, Field<Tscal>(1, computed_fields::SOUNDSPEED, "c_s"));

    // Register common fields in field maps for physics-agnostic VTK output
    // Note: density field is registered by physics mode via init_fields()
    storage.scalar_fields[computed_fields::PRESSURE]   = storage.pressure;
    storage.scalar_fields[computed_fields::SOUNDSPEED] = storage.soundspeed;

    // Register gradient fields for MUSCL reconstruction
    storage.grad_density = solver_graph.register_edge(
        computed_fields::GRAD_DENSITY,
        Field<Tvec>(1, computed_fields::GRAD_DENSITY, "\\nabla\\rho"));
    storage.grad_pressure = solver_graph.register_edge(
        computed_fields::GRAD_PRESSURE,
        Field<Tvec>(1, computed_fields::GRAD_PRESSURE, "\\nabla P"));
    storage.grad_vx = solver_graph.register_edge(
        computed_fields::GRAD_VX, Field<Tvec>(1, computed_fields::GRAD_VX, "\\nabla v_x"));
    storage.grad_vy = solver_graph.register_edge(
        computed_fields::GRAD_VY, Field<Tvec>(1, computed_fields::GRAD_VY, "\\nabla v_y"));
    storage.grad_vz = solver_graph.register_edge(
        computed_fields::GRAD_VZ, Field<Tvec>(1, computed_fields::GRAD_VZ, "\\nabla v_z"));

    // Register timestep edges for node communication
    solver_graph.register_edge("dt", IDataEdge<Tscal>("dt", "dt"));
    solver_graph.register_edge("dt_half", IDataEdge<Tscal>("dt_half", "\\frac{dt}{2}"));
    solver_graph.register_edge("dt_next", IDataEdge<Tscal>("dt_next", "dt_{\\rm next}"));

    // Register convergence edge for h-iteration
    solver_graph.register_edge("h_converged", IDataEdge<bool>("h_converged", "h_{\\rm conv}"));

    // ═══════════════════════════════════════════════════════════════════════════
    // NODES - Wrap computational steps as graph nodes
    // ═══════════════════════════════════════════════════════════════════════════
    // NOTE: PhysicsMode owns the timestep via evolve_timestep(). These nodes are
    // for shared operations that PhysicsMode calls via SolverCallbacks.
    // ═══════════════════════════════════════════════════════════════════════════
    using modules::FunctorNode;

    // Tree building
    solver_graph.register_node(
        "build_trees",
        FunctorNode("BuildTrees", "Build spatial trees for neighbor search", [this]() {
            modules::BuildTrees<Tvec, Kern>(context, solver_config, storage)
                .build_merged_pos_trees();
        }));

    // Presteps for interaction range
    solver_graph.register_node(
        "compute_presteps", FunctorNode("Presteps", "Compute interaction ranges", [this]() {
            modules::BuildTrees<Tvec, Kern>(context, solver_config, storage)
                .compute_presteps_rint();
        }));

    // Neighbor cache
    solver_graph.register_node(
        "start_neighbors", FunctorNode("Neighbors", "Build neighbor cache", [this]() {
            modules::NeighbourCache<Tvec, Kern>(context, solver_config, storage).build_cache();
        }));

    // Gradients (MUSCL)
    solver_graph.register_node(
        "compute_gradients",
        FunctorNode(
            "Gradients",
            "Compute MUSCL gradients: $\\nabla\\rho$, $\\nabla P$, $\\nabla v$",
            [this]() {
                modules::ComputeGradients<Tvec, Kern>(context, solver_config, storage).compute();
            }));

    // Ghost layout initialization
    solver_graph.register_node(
        "init_ghost_layout", FunctorNode("GhostLayout", "Initialize ghost data layout", [this]() {
            init_ghost_layout();
        }));

    // Ghost field communication
    solver_graph.register_node(
        "communicate_ghosts",
        FunctorNode("CommGhosts", "Communicate and merge ghost fields", [this]() {
            modules::GhostCommunicator<Tvec, Kern>(context, solver_config, storage)
                .communicate_merge_ghosts_fields();
        }));

    // Copy density to patchdata
    solver_graph.register_node(
        "copy_density",
        FunctorNode("CopyDensity", "Copy computed density to particle data", [this]() {
            copy_density_to_patchdata();
        }));

    // CFL timestep computation
    solver_graph.register_node(
        "compute_dt", FunctorNode("CFL", "Compute CFL timestep constraint", [this]() {
            Tscal dt_next
                = modules::ComputeCFL<Tvec, Kern>(context, solver_config, storage).compute();
            Tscal dt = storage.solver_graph.template get_edge_ref<IDataEdge<Tscal>>("dt").data;
            // Limit growth to 2x
            if (dt > Tscal(0)) {
                dt_next = sham::min(dt_next, Tscal(2) * dt);
            }
            storage.solver_graph.template get_edge_ref<IDataEdge<Tscal>>("dt_next").data = dt_next;
        }));

    // Cleanup node
    solver_graph.register_node(
        "cleanup", FunctorNode("Cleanup", "Reset caches for next iteration", [this]() {
            reset_neighbors_cache();
            reset_presteps_rint();
            clear_merged_pos_trees();
            reset_merge_ghosts_fields();
            storage.merged_xyzh.reset();
            clear_ghost_cache();
            reset_serial_patch_tree();
            reset_ghost_handler();
            storage.ghost_layout.reset();
        }));

    // NOTE: PhysicsMode::evolve_timestep() composes operations via SolverCallbacks.
    // No operation sequences needed here - mode owns the timestep flow.
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::vtk_do_dump(
    std::string filename, bool add_patch_world_id) {

    // Get field names from physics mode - physics decides what to output
    std::vector<std::string> physics_fields;
    if (physics_mode) {
        physics_fields = physics_mode->get_output_field_names();
    }

    modules::VTKDump<Tvec, Kern>(context, solver_config, storage)
        .do_dump(filename, add_patch_world_id, physics_fields);
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
void shammodels::gsph::Solver<Tvec, Kern>::clear_merged_pos_trees() {
    StackEntry stack_loc{};
    storage.merged_pos_trees.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_presteps_rint() {
    storage.rtree_rint_field.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_neighbors_cache() {
    storage.neigh_cache->neigh_cache = {};
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

    // Physics mode extends ghost layout with physics-specific fields
    if (physics_mode) {
        physics_mode->extend_ghost_layout(ghost_layout);
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::reset_merge_ghosts_fields() {
    storage.merged_patchdata_ghost.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::copy_density_to_patchdata() {
    StackEntry stack_loc{};

    using namespace shamrock;

    // Copy density from storage.density to main particle data "rho" field
    solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);
    patch::PatchDataLayerLayout &pdl         = scheduler().pdl();
    const u32 irho                           = pdl.get_field_idx<Tscal>("rho");

    scheduler().for_each_patchdata_nonempty([&](const patch::Patch p, patch::PatchDataLayer &pdat) {
        u32 cnt = pdat.get_obj_cnt();
        if (cnt == 0) {
            return;
        }

        // Get source (storage.density) and destination (pdat.rho)
        sham::DeviceBuffer<Tscal> &src_buf = density_field.get(p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &dst_buf = pdat.get_field<Tscal>(irho).get_buf();

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto src = src_buf.get_read_access(depends_list);
        auto dst = dst_buf.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parallel_for(cgh, cnt, "gsph_copy_density", [=](u64 gid) {
                dst[gid] = src[gid];
            });
        });

        src_buf.complete_event_state(e);
        dst_buf.complete_event_state(e);
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::gsph::Solver<Tvec, Kern>::update_sync_load_values() {}

template<class Tvec, template<class> class Kern>
shammodels::gsph::TimestepLog shammodels::gsph::Solver<Tvec, Kern>::evolve_once() {
    using namespace shamrock::solvergraph;

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

    u64 Npart_all = scheduler().get_total_obj_count();

    // Set timestep in graph edges
    storage.solver_graph.template get_edge_ref<IDataEdge<Tscal>>("dt").data      = dt;
    storage.solver_graph.template get_edge_ref<IDataEdge<Tscal>>("dt_half").data = dt / Tscal(2);

    // Initialize physics mode on first call (lazy init after config is set)
    init_physics_mode();

    // ═══════════════════════════════════════════════════════════════════════════
    // PHYSICS MODE OWNS THE TIMESTEP
    // ═══════════════════════════════════════════════════════════════════════════
    // Create callbacks for shared operations that PhysicsMode can invoke.
    // This allows each mode to define its own timestep sequence while
    // reusing common operations (tree build, ghost handling, etc.)
    // ═══════════════════════════════════════════════════════════════════════════

    core::SolverCallbacks<Tscal> callbacks;

    callbacks.gen_serial_patch_tree = [this]() {
        gen_serial_patch_tree();
    };

    callbacks.apply_position_boundary = [this](Tscal t) {
        modules::BoundaryHandler<Tvec, Kern>(context, solver_config, storage)
            .apply_position_boundary(t);
    };

    callbacks.gen_ghost_handler = [this](Tscal t) {
        gen_ghost_handler(t);
    };

    callbacks.build_ghost_cache = [this]() {
        build_ghost_cache();
    };

    callbacks.merge_position_ghost = [this]() {
        merge_position_ghost();
    };

    callbacks.build_trees = [this]() {
        storage.solver_graph.get_node_ptr_base("build_trees")->evaluate();
    };

    callbacks.compute_presteps = [this]() {
        storage.solver_graph.get_node_ptr_base("compute_presteps")->evaluate();
    };

    callbacks.start_neighbors = [this]() {
        storage.solver_graph.get_node_ptr_base("start_neighbors")->evaluate();
    };

    // Note: compute_omega callback is no longer used - each physics mode has its own
    // h-iteration/density implementation (NewtonianMode::compute_omega_newtonian,
    // SRMode::compute_omega_sr)

    callbacks.init_ghost_layout = [this]() {
        storage.solver_graph.get_node_ptr_base("init_ghost_layout")->evaluate();
    };

    callbacks.communicate_ghosts = [this]() {
        storage.solver_graph.get_node_ptr_base("communicate_ghosts")->evaluate();
    };

    callbacks.compute_gradients = [this]() {
        storage.solver_graph.get_node_ptr_base("compute_gradients")->evaluate();
    };

    callbacks.copy_density = [this]() {
        storage.solver_graph.get_node_ptr_base("copy_density")->evaluate();
    };

    callbacks.compute_cfl = [this]() -> Tscal {
        storage.solver_graph.get_node_ptr_base("compute_dt")->evaluate();
        return storage.solver_graph.template get_edge_ref<IDataEdge<Tscal>>("dt_next").data;
    };

    callbacks.reset_for_h_iteration = [this]() {
        reset_neighbors_cache();
        reset_presteps_rint();
        clear_merged_pos_trees();
        storage.merged_xyzh.reset();
        clear_ghost_cache();
        reset_ghost_handler();
    };

    callbacks.cleanup = [this]() {
        storage.solver_graph.get_node_ptr_base("cleanup")->evaluate();
    };

    callbacks.h_max_subcycles = solver_config.h_max_subcycles_count;

    // Execute the physics timestep - mode owns the sequence
    Tscal dt_next
        = physics_mode->evolve_timestep(storage, solver_config, scheduler(), dt, callbacks);

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

// ════════════════════════════════════════════════════════════════════════════
// Physics mode selection implementations
// ════════════════════════════════════════════════════════════════════════════

namespace shammodels::gsph {

    template<class Tvec, template<class> class SPHKernel>
    void Solver<Tvec, SPHKernel>::set_physics_newtonian(bool use_grad_h) {
        physics::NewtonianConfig<Tvec> config;
        config.use_grad_h = use_grad_h;
        physics_mode      = core::PhysicsModeFactory::create_newtonian<Tvec, SPHKernel>(config);
    }

    template<class Tvec, template<class> class SPHKernel>
    void Solver<Tvec, SPHKernel>::set_physics_sr(Tscal c_speed) {
        physics::SRConfig<Tvec> config;
        config.c_speed = c_speed;
        physics_mode   = core::PhysicsModeFactory::create_sr<Tvec, SPHKernel>(config);
    }

    template<class Tvec, template<class> class SPHKernel>
    void Solver<Tvec, SPHKernel>::set_physics_mhd(Tscal resistivity) {
        (void) resistivity;
        shambase::throw_with_loc<std::runtime_error>(
            "MHD physics mode not yet implemented for GSPH");
    }

    template<class Tvec, template<class> class SPHKernel>
    bool Solver<Tvec, SPHKernel>::is_physics_newtonian() const {
        if (!physics_mode)
            return false;
        return physics_mode->name() == "Newtonian";
    }

    template<class Tvec, template<class> class SPHKernel>
    bool Solver<Tvec, SPHKernel>::is_physics_sr() const {
        if (!physics_mode)
            return false;
        return physics_mode->name() == "SpecialRelativistic";
    }

    template<class Tvec, template<class> class SPHKernel>
    bool Solver<Tvec, SPHKernel>::is_physics_mhd() const {
        if (!physics_mode)
            return false;
        return physics_mode->name() == "MHD";
    }

    // Initialize physics mode if not already set (defaults to Newtonian)
    template<class Tvec, template<class> class SPHKernel>
    void Solver<Tvec, SPHKernel>::init_physics_mode() {
        if (!physics_mode) {
            set_physics_newtonian();
        }
        // Initialize physics-specific fields (SR conserved variables, etc.)
        // The init_fields method is idempotent - checks if already initialized
        if (!storage.physics_fields_initialized) {
            physics_mode->init_fields(storage, solver_config);
            storage.physics_fields_initialized = true;
        }
    }

} // namespace shammodels::gsph

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

// Truncated Gaussian kernel (TGauss3) - for SR-GSPH
template class shammodels::gsph::Solver<f64_3, TGauss3>;
