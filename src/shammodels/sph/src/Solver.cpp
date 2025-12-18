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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shambase/memory.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/reduction.hpp"
#include "shambackends/MemPerfInfos.hpp"
#include "shambackends/details/memoryHandle.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/timestep_report.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/math/q_ab.hpp"
#include "shammodels/sph/modules/BuildTrees.hpp"
#include "shammodels/sph/modules/ComputeEos.hpp"
#include "shammodels/sph/modules/ComputeLoadBalanceValue.hpp"
#include "shammodels/sph/modules/ComputeNeighStats.hpp"
#include "shammodels/sph/modules/ComputeOmega.hpp"
#include "shammodels/sph/modules/ConservativeCheck.hpp"
#include "shammodels/sph/modules/DiffOperator.hpp"
#include "shammodels/sph/modules/DiffOperatorDtDivv.hpp"
#include "shammodels/sph/modules/ExternalForces.hpp"
#include "shammodels/sph/modules/GetParticlesOutsideSphere.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensityNeighLim.hpp"
#include "shammodels/sph/modules/IterateSmoothingNumDensity.hpp"
#include "shammodels/sph/modules/KillParticles.hpp"
#include "shammodels/sph/modules/LoopSmoothingLengthIter.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shammodels/sph/modules/ParticleReordering.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shammodels/sph/modules/UpdateDerivs.hpp"
#include "shammodels/sph/modules/UpdateViscosity.hpp"
#include "shammodels/sph/modules/io/VTKDump.hpp"
#include "shammodels/sph/modules/self_gravity/SGDirectPlummer.hpp"
#include "shammodels/sph/modules/self_gravity/SGFMMPlummer.hpp"
#include "shammodels/sph/modules/self_gravity/SGMMPlummer.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamphys/mhd.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/solvergraph/CopyPatchDataFieldFromLayer.hpp"
#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/NodeSetEdge.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::init_solver_graph() {

    storage.part_counts
        = std::make_shared<shamrock::solvergraph::Indexes<u32>>("part_counts", "N_{\\rm part}");

    storage.part_counts_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        "part_counts_with_ghost", "N_{\\rm part, with ghost}");

    storage.patch_rank_owner
        = std::make_shared<shamrock::solvergraph::ScalarsEdge<u32>>("patch_rank_owner", "rank");

    // merged ghost spans
    storage.positions_with_ghosts
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("part_pos", "\\mathbf{r}");
    storage.hpart_with_ghosts
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_part", "h");

    storage.neigh_cache
        = std::make_shared<shammodels::sph::solvergraph::NeighCache>("neigh_cache", "neigh");

    storage.omega = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "omega", "\\Omega");

    if (solver_config.has_field_alphaAV()) {
        storage.alpha_av_updated = std::make_shared<shamrock::solvergraph::Field<Tscal>>(
            1, "alpha_av_updated", "\\alpha_{\\rm AV}");
    }

    storage.pressure = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "pressure", "P");
    storage.soundspeed
        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "soundspeed", "c_s");
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::vtk_do_dump(
    std::string filename, bool add_patch_world_id) {

    modules::VTKDump(context, solver_config).do_dump(filename, add_patch_world_id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Debug interface dump
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace shammodels::sph {

    template<class Tvec>
    struct Debug_ph_dump {
        using Tscal = shambase::VecComponent<Tvec>;

        u64 nobj;
        f64 gpart_mass;

        sycl::buffer<Tvec> &buf_xyz;
        sycl::buffer<Tscal> &buf_hpart;
        sycl::buffer<Tvec> &buf_vxyz;
    };

    template<class Tvec>
    void fill_blocks(PhantomDumpBlock &block, Debug_ph_dump<Tvec> &info) {

        using Tscal           = shambase::VecComponent<Tvec>;
        std::vector<Tvec> xyz = shamalgs::memory::buf_to_vec(info.buf_xyz, info.nobj);

        u64 xid = block.get_ref_fort_real("x");
        u64 yid = block.get_ref_fort_real("y");
        u64 zid = block.get_ref_fort_real("z");

        for (auto vec : xyz) {
            block.blocks_fort_real[xid].vals.push_back(vec.x());
            block.blocks_fort_real[yid].vals.push_back(vec.y());
            block.blocks_fort_real[zid].vals.push_back(vec.z());
        }

        std::vector<Tscal> h = shamalgs::memory::buf_to_vec(info.buf_hpart, info.nobj);
        u64 hid              = block.get_ref_f32("h");
        for (auto h_ : h) {
            block.blocks_f32[hid].vals.push_back(h_);
        }

        std::vector<Tvec> vxyz = shamalgs::memory::buf_to_vec(info.buf_vxyz, info.nobj);

        u64 vxid = block.get_ref_fort_real("vx");
        u64 vyid = block.get_ref_fort_real("vy");
        u64 vzid = block.get_ref_fort_real("vz");

        for (auto vec : vxyz) {
            block.blocks_fort_real[vxid].vals.push_back(vec.x());
            block.blocks_fort_real[vyid].vals.push_back(vec.y());
            block.blocks_fort_real[vzid].vals.push_back(vec.z());
        }

        block.tot_count = block.blocks_fort_real[xid].vals.size();
    }

    template<class Tvec>
    shammodels::sph::PhantomDump make_interface_debug_phantom_dump(Debug_ph_dump<Tvec> info) {

        using Tscal = shambase::VecComponent<Tvec>;
        PhantomDump dump;

        dump.override_magic_number();
        dump.iversion = 1;
        dump.fileid   = shambase::format("{:100s}", "FT:Phantom Shamrock writter");

        u32 Ntot = info.nobj;
        dump.table_header_fort_int.add("nparttot", Ntot);
        dump.table_header_fort_int.add("ntypes", 8);
        dump.table_header_fort_int.add("npartoftype", Ntot);
        dump.table_header_fort_int.add("npartoftype", 0);
        dump.table_header_fort_int.add("npartoftype", 0);
        dump.table_header_fort_int.add("npartoftype", 0);
        dump.table_header_fort_int.add("npartoftype", 0);
        dump.table_header_fort_int.add("npartoftype", 0);
        dump.table_header_fort_int.add("npartoftype", 0);
        dump.table_header_fort_int.add("npartoftype", 0);

        dump.table_header_i64.add("nparttot", Ntot);
        dump.table_header_i64.add("ntypes", 8);
        dump.table_header_i64.add("npartoftype", Ntot);
        dump.table_header_i64.add("npartoftype", 0);
        dump.table_header_i64.add("npartoftype", 0);
        dump.table_header_i64.add("npartoftype", 0);
        dump.table_header_i64.add("npartoftype", 0);
        dump.table_header_i64.add("npartoftype", 0);
        dump.table_header_i64.add("npartoftype", 0);
        dump.table_header_i64.add("npartoftype", 0);

        dump.table_header_fort_int.add("nblocks", 1);
        dump.table_header_fort_int.add("nptmass", 0);
        dump.table_header_fort_int.add("ndustlarge", 0);
        dump.table_header_fort_int.add("ndustsmall", 0);
        dump.table_header_fort_int.add("idust", 7);
        dump.table_header_fort_int.add("idtmax_n", 1);
        dump.table_header_fort_int.add("idtmax_frac", 0);
        dump.table_header_fort_int.add("idumpfile", 0);
        dump.table_header_fort_int.add("majorv", 2023);
        dump.table_header_fort_int.add("minorv", 0);
        dump.table_header_fort_int.add("microv", 0);
        dump.table_header_fort_int.add("isink", 0);

        dump.table_header_i32.add("iexternalforce", 0);
        dump.table_header_i32.add("ieos", 2);
        dump.table_header_fort_real.add("gamma", 1.66667);
        dump.table_header_fort_real.add("RK2", 0);
        dump.table_header_fort_real.add("polyk2", 0);
        dump.table_header_fort_real.add("qfacdisc", 0.75);
        dump.table_header_fort_real.add("qfacdisc2", 0.75);

        dump.table_header_fort_real.add("time", 0);
        dump.table_header_fort_real.add("dtmax", 0.1);

        dump.table_header_fort_real.add("rhozero", 0);
        dump.table_header_fort_real.add("hfact", 1.2);
        dump.table_header_fort_real.add("tolh", 0.0001);
        dump.table_header_fort_real.add("C_cour", 0);
        dump.table_header_fort_real.add("C_force", 0);
        dump.table_header_fort_real.add("alpha", 0);
        dump.table_header_fort_real.add("alphau", 1);
        dump.table_header_fort_real.add("alphaB", 1);

        dump.table_header_fort_real.add("massoftype", info.gpart_mass);
        dump.table_header_fort_real.add("massoftype", 0);
        dump.table_header_fort_real.add("massoftype", 0);
        dump.table_header_fort_real.add("massoftype", 0);
        dump.table_header_fort_real.add("massoftype", 0);
        dump.table_header_fort_real.add("massoftype", 0);
        dump.table_header_fort_real.add("massoftype", 0);
        dump.table_header_fort_real.add("massoftype", 0);

        dump.table_header_fort_real.add("Bextx", 0);
        dump.table_header_fort_real.add("Bexty", 0);
        dump.table_header_fort_real.add("Bextz", 0);
        dump.table_header_fort_real.add("dum", 0);

        dump.table_header_fort_real.add("get_conserv", -1);
        dump.table_header_fort_real.add("etot_in", 0.59762);
        dump.table_header_fort_real.add("angtot_in", 0.0189694);
        dump.table_header_fort_real.add("totmom_in", 0.0306284);

        dump.table_header_f64.add("udist", 1);
        dump.table_header_f64.add("umass", 1);
        dump.table_header_f64.add("utime", 1);
        dump.table_header_f64.add("umagfd", 3.54491);

        PhantomDumpBlock block_part;

        fill_blocks(block_part, info);

        dump.blocks.push_back(std::move(block_part));

        return dump;
    }

} // namespace shammodels::sph

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::gen_serial_patch_tree() {
    StackEntry stack_loc{};

    SerialPatchTree<Tvec> _sptree = SerialPatchTree<Tvec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));
}

/**
 * @brief Applies position boundary conditions to the particles.
 *
 * @param time_val the current time value
 */
template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::apply_position_boundary(Tscal time_val) {

    StackEntry stack_loc{};

    shamlog_debug_ln("SphSolver", "apply position boundary");

    PatchScheduler &sched = scheduler();

    shamrock::SchedulerUtility integrators(sched);
    shamrock::ReattributeDataUtility reatrib(sched);

    auto &pdl = sched.pdl();

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
        integrators.fields_apply_shearing_periodicity(
            ixyz,
            ivxyz,
            std::pair{bmin, bmax},
            c->shear_base,
            c->shear_dir,
            c->shear_speed * time_val,
            c->shear_speed);
    }

    reatrib.reatribute_patch_objects(storage.serial_patch_tree.get(), "xyz");
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::build_ghost_cache() {

    StackEntry stack_loc{};

    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;
    SPHUtils sph_utils(scheduler());

    storage.ghost_patch_cache.set(sph_utils.build_interf_cache(
        storage.ghost_handler.get(),
        storage.serial_patch_tree.get(),
        solver_config.htol_up_coarse_cycle));

    // storage.ghost_handler.get().gen_debug_patch_ghost(storage.ghost_patch_cache.get());
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::clear_ghost_cache() {
    StackEntry stack_loc{};
    storage.ghost_patch_cache.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::merge_position_ghost() {

    StackEntry stack_loc{};

    storage.merged_xyzh.set(
        storage.ghost_handler.get().build_comm_merge_positions(storage.ghost_patch_cache.get()));

    { // set element counts
        shambase::get_check_ref(storage.part_counts).indexes
            = storage.merged_xyzh.get().template map<u32>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return scheduler().patch_data.get_pdat(id).get_obj_cnt();
                });
    }

    { // set element counts
        shambase::get_check_ref(storage.part_counts_with_ghost).indexes
            = storage.merged_xyzh.get().template map<u32>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return mpdat.get_obj_cnt();
                });
    }

    { // Attach spans to block coords
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
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::build_merged_pos_trees() {
    modules::BuildTrees(context, solver_config, storage).build_merged_pos_trees();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::clear_merged_pos_trees() {
    StackEntry stack_loc{};
    storage.merged_pos_trees.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::do_predictor_leapfrog(Tscal dt) {

    StackEntry stack_loc{};
    using namespace shamrock::patch;
    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint           = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint          = pdl.get_field_idx<Tscal>("duint");

    bool has_B_field       = solver_config.has_field_B_on_rho();
    bool has_psi_field     = solver_config.has_field_psi_on_ch();
    bool has_epsilon_field = solver_config.dust_config.has_epsilon_field();
    bool has_deltav_field  = solver_config.dust_config.has_deltav_field();

    const u32 iB_on_rho   = (has_B_field) ? pdl.get_field_idx<Tvec>("B/rho") : 0;
    const u32 idB_on_rho  = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : 0;
    const u32 ipsi_on_ch  = (has_psi_field) ? pdl.get_field_idx<Tscal>("psi/ch") : 0;
    const u32 idpsi_on_ch = (has_psi_field) ? pdl.get_field_idx<Tscal>("dpsi/ch") : 0;

    const u32 iepsilon   = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("epsilon") : 0;
    const u32 idtepsilon = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("dtepsilon") : 0;
    const u32 ideltav    = (has_deltav_field) ? pdl.get_field_idx<Tvec>("deltav") : 0;
    const u32 idtdeltav  = (has_deltav_field) ? pdl.get_field_idx<Tvec>("dtdeltav") : 0;

    shamrock::SchedulerUtility utility(scheduler());

    // forward euler step f dt/2
    shamlog_debug_ln("sph::BasicGas", "forward euler step f dt/2");
    utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
    utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

    if (has_B_field) {
        utility.fields_forward_euler<Tvec>(iB_on_rho, idB_on_rho, dt / 2);
    }
    if (has_psi_field) {
        utility.fields_forward_euler<Tscal>(ipsi_on_ch, idpsi_on_ch, dt / 2);
    }

    // forward euler step positions dt
    shamlog_debug_ln("sph::BasicGas", "forward euler step positions dt");
    utility.fields_forward_euler<Tvec>(ixyz, ivxyz, dt);

    // forward euler step f dt/2
    shamlog_debug_ln("sph::BasicGas", "forward euler step f dt/2");
    utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
    utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

    if (has_B_field) {
        utility.fields_forward_euler<Tvec>(iB_on_rho, idB_on_rho, dt / 2);
    }
    if (has_psi_field) {
        utility.fields_forward_euler<Tscal>(ipsi_on_ch, idpsi_on_ch, dt / 2);
    }
    if (has_epsilon_field) {
        utility.fields_forward_euler<Tscal>(iepsilon, idtepsilon, dt / 2);
    }
    if (has_deltav_field) {
        utility.fields_forward_euler<Tvec>(ideltav, idtdeltav, dt / 2);
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::sph_prestep(Tscal time_val, Tscal dt) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    using RTree    = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;
    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;

    SPHUtils sph_utils(scheduler());
    shamrock::SchedulerUtility utility(scheduler());

    PatchDataLayerLayout &pdl = scheduler().pdl();
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
        start_neighbors_cache(time_val);

        _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1, Tscal(100));
        _h_old     = utility.save_field<Tscal>(ihpart, "h_old");

        Tscal max_eps_h;

        if (solver_config.gpart_mass == 0) {
            shambase::throw_with_loc<std::runtime_error>(shambase::format(
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

        using h_conf_density_based     = typename SmoothingLengthConfig::DensityBased;
        using h_conf_neigh_lim         = typename SmoothingLengthConfig::DensityBasedNeighLim;
        using h_conf_num_density_based = typename SmoothingLengthConfig::NumDensityBased;
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
        } else if (
            h_conf_num_density_based *conf = std::get_if<h_conf_num_density_based>(
                &solver_config.smoothing_length_config.config)) {

            if (conf->mode == h_conf_num_density_based::Mode::BaseKernel) {
                std::shared_ptr<
                    shammodels::sph::modules::IterateSmoothingNumDensity<Tvec, shammath::M4<Tscal>>>
                    smth_h_iter_num_density
                    = std::make_shared<shammodels::sph::modules::
                                           IterateSmoothingNumDensity<Tvec, shammath::M4<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::DoubleHump) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4DH<Tscal>>>
                    smth_h_iter_num_density
                    = std::make_shared<shammodels::sph::modules::
                                           IterateSmoothingNumDensity<Tvec, shammath::M4DH<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::DoubleHump3) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4DH3<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4DH3<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::DoubleHump5) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4DH5<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4DH5<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::DoubleHump7) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4DH7<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4DH7<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::Shift2) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4Shift2<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4Shift2<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::Shift4) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4Shift4<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4Shift4<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::Shift8) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4Shift8<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4Shift8<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            } else if (conf->mode == h_conf_num_density_based::Mode::Shift16) {
                std::shared_ptr<shammodels::sph::modules::
                                    IterateSmoothingNumDensity<Tvec, shammath::M4Shift16<Tscal>>>
                    smth_h_iter_num_density = std::make_shared<
                        shammodels::sph::modules::
                            IterateSmoothingNumDensity<Tvec, shammath::M4Shift16<Tscal>>>(
                        solver_config.gpart_mass,
                        solver_config.htol_up_coarse_cycle,
                        solver_config.htol_up_fine_cycle);
                smth_h_iter_num_density->set_edges(
                    sizes, neigh_cache, pos_merged, hold, hnew, eps_h, should_set_omega_mask);
                smth_h_iter_ptr = smth_h_iter_num_density;
            }

            else {
                throw std::runtime_error("Invalid smoothing length configuration");
            }

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
                                add_info += shambase::format(
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
        logger::err_ln("SPH", "the h iterator is not converged after", hstep_cnt, "iterations");
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hnew_edge
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
    shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};
    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        auto &field = pdat.get_field<Tscal>(ihpart);
        hnew_refs.add_obj(p.id_patch, std::ref(field));
    });
    hnew_edge->set_refs(hnew_refs);

    modules::NodeComputeOmega<Tvec, Kern> compute_omega{solver_config.gpart_mass};
    compute_omega.set_edges(
        storage.part_counts,
        storage.neigh_cache,
        storage.positions_with_ghosts,
        hnew_edge,
        storage.omega);
    compute_omega.evaluate();

    if (solver_config.smoothing_length_config.is_density_based_neigh_lim()
        || solver_config.smoothing_length_config.is_num_density_based()) {
        // if the h limiter is triggered, omega does not hold it's sense of dh/dr anymore
        // so we set it to 1, this effectively is equivalent of disabling the energy correction
        // term corresponding to dh/dr
        modules::SetWhenMask<Tscal> set_omega_mask{1};
        set_omega_mask.set_edges(storage.part_counts, should_set_omega_mask, storage.omega);
        set_omega_mask.evaluate();
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::init_ghost_layout() {

    storage.ghost_layout.set(std::make_shared<shamrock::patch::PatchDataLayerLayout>());

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());

    solver_config.set_ghost_layout(ghost_layout);
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::compute_presteps_rint() {

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

                // the old tree used to increase the size of the hmax of the tree nodes by the
                // tolerance so we do it also with the new tree, maybe we should move that somewhere
                // else.
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
void shammodels::sph::Solver<Tvec, Kern>::reset_presteps_rint() {
    storage.rtree_rint_field.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::start_neighbors_cache(f64 t) {
    if (solver_config.use_two_stage_search) {
        shammodels::sph::modules::NeighbourCache<Tvec, u_morton, Kern>(
            context, solver_config, storage)
            .start_neighbors_cache_2stages();
    } else {
        shammodels::sph::modules::NeighbourCache<Tvec, u_morton, Kern>(
            context, solver_config, storage)
            .start_neighbors_cache();
    }

    if (solver_config.show_neigh_stats) {
        auto &pos_merged        = storage.positions_with_ghosts;
        auto &neigh_cache       = storage.neigh_cache;
        auto &hpart_with_ghosts = storage.hpart_with_ghosts;
        auto &part_counts       = storage.part_counts;

        std::shared_ptr<shamrock::solvergraph::IDataEdge<f64>> sim_time
            = std::make_shared<shamrock::solvergraph::IDataEdge<f64>>("sim_time", "sim_time");
        sim_time->data = t;

        modules::ComputeNeighStats<Tvec> compute_neigh_stats(
            Kernel::Rkern, solver_config.neigh_stats_filename);

        compute_neigh_stats.set_edges(
            sim_time, part_counts, neigh_cache, pos_merged, hpart_with_ghosts);
        compute_neigh_stats.evaluate();
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::reset_neighbors_cache() {
    // storage.neighbors_cache.reset();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::communicate_merge_ghosts_fields() {

    StackEntry stack_loc{};

    shambase::Timer timer_interf;
    timer_interf.start();

    using namespace shamrock;
    using namespace shamrock::patch;

    bool has_alphaAV_field    = solver_config.has_field_alphaAV();
    bool has_soundspeed_field = solver_config.ghost_has_soundspeed();

    bool has_B_field       = solver_config.has_field_B_on_rho();
    bool has_psi_field     = solver_config.has_field_psi_on_ch();
    bool has_curlB_field   = solver_config.has_field_curlB();
    bool has_epsilon_field = solver_config.dust_config.has_epsilon_field();
    bool has_deltav_field  = solver_config.dust_config.has_deltav_field();

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint           = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint          = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");

    const u32 ialpha_AV   = (has_alphaAV_field) ? pdl.get_field_idx<Tscal>("alpha_AV") : 0;
    const u32 isoundspeed = (has_soundspeed_field) ? pdl.get_field_idx<Tscal>("soundspeed") : 0;

    const u32 iB_on_rho   = (has_B_field) ? pdl.get_field_idx<Tvec>("B/rho") : 0;
    const u32 idB_on_rho  = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : 0;
    const u32 ipsi_on_ch  = (has_psi_field) ? pdl.get_field_idx<Tscal>("psi/ch") : 0;
    const u32 idpsi_on_ch = (has_psi_field) ? pdl.get_field_idx<Tscal>("dpsi/ch") : 0;
    const u32 icurlB      = (has_curlB_field) ? pdl.get_field_idx<Tvec>("curlB") : 0;

    bool do_MHD_debug       = solver_config.do_MHD_debug();
    const u32 imag_pressure = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("mag_pressure") : -1;
    const u32 imag_tension  = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("mag_tension") : -1;
    const u32 igas_pressure = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("gas_pressure") : -1;
    const u32 itensile_corr = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("tensile_corr") : -1;
    const u32 ipsi_propag   = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("psi_propag") : -1;
    const u32 ipsi_diff     = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("psi_diff") : -1;
    const u32 ipsi_cons     = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("psi_cons") : -1;
    const u32 iu_mhd        = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("u_mhd") : -1;

    const u32 iepsilon = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("epsilon") : 0;
    const u32 ideltav  = (has_deltav_field) ? pdl.get_field_idx<Tvec>("deltav") : 0;

    auto ghost_layout_ptr                               = storage.ghost_layout.get();
    shamrock::patch::PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(ghost_layout_ptr);
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf  = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 iaxyz_interf
        = (solver_config.has_axyz_in_ghost()) ? ghost_layout.get_field_idx<Tvec>("axyz") : 0;

    const u32 isoundspeed_interf
        = (has_soundspeed_field) ? ghost_layout.get_field_idx<Tscal>("soundspeed") : 0;

    const u32 iB_interf     = (has_B_field) ? ghost_layout.get_field_idx<Tvec>("B/rho") : 0;
    const u32 ipsi_interf   = (has_psi_field) ? ghost_layout.get_field_idx<Tscal>("psi/ch") : 0;
    const u32 icurlB_interf = (has_curlB_field) ? ghost_layout.get_field_idx<Tvec>("curlB") : 0;

    const u32 iepsilon_interf
        = (has_epsilon_field) ? ghost_layout.get_field_idx<Tscal>("epsilon") : 0;
    const u32 ideltav_interf = (has_deltav_field) ? ghost_layout.get_field_idx<Tvec>("deltav") : 0;

    using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
    shamrock::solvergraph::Field<Tscal> &omega    = shambase::get_check_ref(storage.omega);

    auto pdat_interf = ghost_handle.template build_interface_native<PatchDataLayer>(
        storage.ghost_patch_cache.get(),
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sham::DeviceBuffer<u32> &buf_idx, u32 cnt) {
            PatchDataLayer pdat(ghost_layout_ptr);

            pdat.reserve(cnt);

            return pdat;
        });

    ghost_handle.template modify_interface_native<PatchDataLayer>(
        storage.ghost_patch_cache.get(),
        pdat_interf,
        [&](u64 sender,
            u64,
            InterfaceBuildInfos binfo,
            sham::DeviceBuffer<u32> &buf_idx,
            u32 cnt,
            PatchDataLayer &pdat) {
            PatchDataLayer &sender_patch        = scheduler().patch_data.get_pdat(sender);
            PatchDataField<Tscal> &sender_omega = omega.get(sender);

            sender_patch.get_field<Tscal>(ihpart).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(ihpart_interf));
            sender_patch.get_field<Tscal>(iuint).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(iuint_interf));

            if (solver_config.has_axyz_in_ghost()) {
                sender_patch.get_field<Tvec>(iaxyz).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(iaxyz_interf));
            }

            sender_patch.get_field<Tvec>(ivxyz).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tvec>(ivxyz_interf));

            sender_omega.append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(iomega_interf));

            if (has_soundspeed_field) {
                sender_patch.get_field<Tscal>(isoundspeed)
                    .append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(isoundspeed_interf));
            }

            if (has_B_field) {
                sender_patch.get_field<Tvec>(iB_on_rho).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(iB_interf));
            }

            if (has_psi_field) {
                sender_patch.get_field<Tscal>(ipsi_on_ch)
                    .append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(ipsi_interf));
            }

            if (has_curlB_field) {
                sender_patch.get_field<Tvec>(icurlB).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(icurlB_interf));
            }

            if (has_epsilon_field) {
                sender_patch.get_field<Tscal>(iepsilon).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(iepsilon_interf));
            }

            if (has_deltav_field) {
                sender_patch.get_field<Tvec>(ideltav).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tvec>(ideltav_interf));
            }
        });

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

    shambase::DistributedDataShared<PatchDataLayer> interf_pdat
        = ghost_handle.communicate_pdat(ghost_layout_ptr, std::move(pdat_interf));

    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchDataLayer &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    storage.merged_patchdata_ghost.set(
        ghost_handle.template merge_native<PatchDataLayer, PatchDataLayer>(
            std::move(interf_pdat),
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                PatchDataLayer pdat_new(ghost_layout_ptr);

                u32 or_elem = pdat.get_obj_cnt();
                pdat_new.reserve(or_elem + sz_interf_map[p.id_patch]);
                u32 total_elements = or_elem;

                PatchDataField<Tscal> &cur_omega = omega.get(p.id_patch);

                pdat_new.get_field<Tscal>(ihpart_interf).insert(pdat.get_field<Tscal>(ihpart));
                pdat_new.get_field<Tscal>(iuint_interf).insert(pdat.get_field<Tscal>(iuint));
                pdat_new.get_field<Tvec>(ivxyz_interf).insert(pdat.get_field<Tvec>(ivxyz));

                if (solver_config.has_axyz_in_ghost()) {
                    pdat_new.get_field<Tvec>(iaxyz_interf).insert(pdat.get_field<Tvec>(iaxyz));
                }

                pdat_new.get_field<Tscal>(iomega_interf).insert(cur_omega);

                if (has_soundspeed_field) {
                    pdat_new.get_field<Tscal>(isoundspeed_interf)
                        .insert(pdat.get_field<Tscal>(isoundspeed));
                }

                if (has_B_field) {
                    pdat_new.get_field<Tvec>(iB_interf).insert(pdat.get_field<Tvec>(iB_on_rho));
                }

                if (has_psi_field) {
                    pdat_new.get_field<Tscal>(ipsi_interf)
                        .insert(pdat.get_field<Tscal>(ipsi_on_ch));
                }

                if (has_curlB_field) {
                    pdat_new.get_field<Tvec>(icurlB_interf).insert(pdat.get_field<Tvec>(icurlB));
                }

                if (has_epsilon_field) {
                    pdat_new.get_field<Tscal>(iepsilon_interf)
                        .insert(pdat.get_field<Tscal>(iepsilon));
                }

                if (has_deltav_field) {
                    pdat_new.get_field<Tvec>(ideltav_interf).insert(pdat.get_field<Tvec>(ideltav));
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
void shammodels::sph::Solver<Tvec, Kern>::reset_merge_ghosts_fields() {
    storage.merged_patchdata_ghost.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// start artificial viscosity section //////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::update_artificial_viscosity(Tscal dt) {

    sph::modules::UpdateViscosity<Tvec, Kern>(context, solver_config, storage)
        .update_artificial_viscosity(dt);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// end artificial viscosity section ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::compute_eos_fields() {

    modules::ComputeEos<Tvec, Kern>(context, solver_config, storage).compute_eos();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::reset_eos_fields() {
    shambase::get_check_ref(storage.pressure).free_alloc();
    shambase::get_check_ref(storage.soundspeed).free_alloc();
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::prepare_corrector() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;
    shamrock::SchedulerUtility utility(scheduler());
    PatchDataLayerLayout &pdl = scheduler().pdl();

    bool has_B_field       = solver_config.has_field_B_on_rho();
    bool has_psi_field     = solver_config.has_field_psi_on_ch();
    bool has_epsilon_field = solver_config.dust_config.has_epsilon_field();
    bool has_deltav_field  = solver_config.dust_config.has_deltav_field();

    const u32 iduint      = pdl.get_field_idx<Tscal>("duint");
    const u32 iaxyz       = pdl.get_field_idx<Tvec>("axyz");
    const u32 idB_on_rho  = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : 0;
    const u32 idpsi_on_ch = (has_psi_field) ? pdl.get_field_idx<Tscal>("dpsi/ch") : 0;

    shamlog_debug_ln("sph::BasicGas", "save old fields");
    storage.old_axyz.set(utility.save_field<Tvec>(iaxyz, "axyz_old"));
    storage.old_duint.set(utility.save_field<Tscal>(iduint, "duint_old"));

    if (has_B_field) {
        storage.old_dB_on_rho.set(utility.save_field<Tvec>(idB_on_rho, "dB/rho_old"));
    }
    if (has_psi_field) {
        storage.old_dpsi_on_ch.set(utility.save_field<Tscal>(idpsi_on_ch, "dpsi/ch_old"));
    }
    if (has_epsilon_field) {
        storage.old_dtepsilon.set(
            utility.save_field<Tscal>(pdl.get_field_idx<Tscal>("dtepsilon"), "dtepsilon_old"));
    }
    if (has_deltav_field) {
        storage.old_dtdeltav.set(
            utility.save_field<Tvec>(pdl.get_field_idx<Tvec>("dtdeltav"), "dtdeltav_old"));
    }
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::update_derivs() {

    modules::UpdateDerivs<Tvec, Kern> derivs(context, solver_config, storage);
    derivs.update_derivs();

    modules::ExternalForces<Tvec, Kern> ext_forces(context, solver_config, storage);
    ext_forces.add_ext_forces();
}

template<class Tvec, template<class> class Kern>
bool shammodels::sph::Solver<Tvec, Kern>::apply_corrector(Tscal dt, u64 Npart_all) {
    return false;
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::update_sync_load_values() {
    modules::ComputeLoadBalanceValue<Tvec, Kern>(context, solver_config, storage)
        .update_load_balancing();
    scheduler().scheduler_step(false, false);

    // give to the solvergraph the patch rank owners
    storage.patch_rank_owner->values = {};
    scheduler().for_each_global_patch([&](const shamrock::patch::Patch p) {
        storage.patch_rank_owner->values.add_obj(
            p.id_patch, scheduler().get_patch_rank_owner(p.id_patch));
    });
}

template<class Tvec, template<class> class Kern>
void shammodels::sph::Solver<Tvec, Kern>::part_killing_step() {

    using namespace shamrock;
    using namespace shamrock::patch;

    if (solver_config.particle_killing.kill_list.size() == 0) {
        return;
    }

    PatchDataLayerLayout &pdl = scheduler().pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> pos
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("", "");

    {
        shamrock::solvergraph::DDPatchDataFieldRef<Tvec> refs = {};
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            refs.add_obj(cur_p.id_patch, std::ref(pdat.get_field<Tvec>(ixyz)));
        });
        pos->set_refs(refs);
    }

    std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> patchdatas
        = std::make_shared<shamrock::solvergraph::PatchDataLayerRefs>("", "");

    {
        patchdatas->free_alloc();
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            patchdatas->patchdatas.add_obj(cur_p.id_patch, std::ref(pdat));
        });
    }

    std::shared_ptr<shamrock::solvergraph::DistributedBuffers<u32>> part_to_remove
        = std::make_shared<shamrock::solvergraph::DistributedBuffers<u32>>("", "");

    std::vector<std::shared_ptr<shamrock::solvergraph::INode>> part_kill_sequence;

    using kill_t      = typename ParticleKillingConfig<Tvec>::kill_t;
    using kill_sphere = typename ParticleKillingConfig<Tvec>::Sphere;

    // selectors
    for (kill_t &kill_obj : solver_config.particle_killing.kill_list) {
        if (kill_sphere *kill_info = std::get_if<kill_sphere>(&kill_obj)) {

            modules::GetParticlesOutsideSphere<Tvec> node_selector(
                kill_info->center, kill_info->radius);
            node_selector.set_edges(pos, part_to_remove);

            part_kill_sequence.push_back(
                std::make_shared<decltype(node_selector)>(std::move(node_selector)));
        }
    }

    { // killing
        modules::KillParticles node_killer{};
        node_killer.set_edges(part_to_remove, patchdatas);

        part_kill_sequence.push_back(
            std::make_shared<decltype(node_killer)>(std::move(node_killer)));
    }

    shamrock::solvergraph::OperationSequence seq("particle killing", std::move(part_kill_sequence));
    seq.evaluate();
}

template<class Tvec, template<class> class Kern>
shammodels::sph::TimestepLog shammodels::sph::Solver<Tvec, Kern>::evolve_once() {

    sham::MemPerfInfos mem_perf_infos_start = sham::details::get_mem_perf_info();
    f64 mpi_timer_start                     = shamcomm::mpi::get_timer("total");

    Tscal t_current = solver_config.get_time();
    Tscal dt        = solver_config.get_dt_sph();

    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        shamcomm::logs::raw_ln(
            shambase::format("---------------- t = {}, dt = {} ----------------", t_current, dt));
    }

    shambase::Timer tstep;
    tstep.start();

    // if(shamcomm::world_rank() == 0) std::cout << scheduler().dump_status() << std::endl;
    modules::ComputeLoadBalanceValue<Tvec, Kern>(context, solver_config, storage)
        .update_load_balancing();
    scheduler().scheduler_step(true, true);
    modules::ComputeLoadBalanceValue<Tvec, Kern>(context, solver_config, storage)
        .update_load_balancing();
    // if(shamcomm::world_rank() == 0) std::cout << scheduler().dump_status() << std::endl;
    scheduler().scheduler_step(false, false);
    // if(shamcomm::world_rank() == 0) std::cout << scheduler().dump_status() << std::endl;

    // give to the solvergraph the patch rank owners
    storage.patch_rank_owner->values = {};
    scheduler().for_each_global_patch([&](const shamrock::patch::Patch p) {
        storage.patch_rank_owner->values.add_obj(
            p.id_patch, scheduler().get_patch_rank_owner(p.id_patch));
    });

    using namespace shamrock;
    using namespace shamrock::patch;

    bool has_B_field       = solver_config.has_field_B_on_rho();
    bool has_psi_field     = solver_config.has_field_psi_on_ch();
    bool has_epsilon_field = solver_config.dust_config.has_epsilon_field();
    bool has_deltav_field  = solver_config.dust_config.has_deltav_field();

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz        = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz       = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz       = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint       = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint      = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart      = pdl.get_field_idx<Tscal>("hpart");
    const u32 iB_on_rho   = (has_B_field) ? pdl.get_field_idx<Tvec>("B/rho") : 0;
    const u32 idB_on_rho  = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : 0;
    const u32 ipsi_on_ch  = (has_psi_field) ? pdl.get_field_idx<Tscal>("psi/ch") : 0;
    const u32 idpsi_on_ch = (has_psi_field) ? pdl.get_field_idx<Tscal>("dpsi/ch") : 0;
    const u32 iepsilon    = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("epsilon") : 0;
    const u32 idtepsilon  = (has_epsilon_field) ? pdl.get_field_idx<Tscal>("dtepsilon") : 0;
    const u32 ideltav     = (has_deltav_field) ? pdl.get_field_idx<Tvec>("deltav") : 0;
    const u32 idtdeltav   = (has_deltav_field) ? pdl.get_field_idx<Tvec>("dtdeltav") : 0;

    shamrock::SchedulerUtility utility(scheduler());

    modules::SinkParticlesUpdate<Tvec, Kern> sink_update(context, solver_config, storage);
    modules::ExternalForces<Tvec, Kern> ext_forces(context, solver_config, storage);

    sink_update.accrete_particles();
    ext_forces.point_mass_accrete_particles();

    do_predictor_leapfrog(dt);

    sink_update.predictor_step(dt);

    part_killing_step();

    sink_update.compute_ext_forces();

    ext_forces.compute_ext_forces_indep_v();

    gen_serial_patch_tree();

    apply_position_boundary(t_current + dt);

    u64 Npart_all = scheduler().get_total_obj_count();

    if (solver_config.enable_particle_reordering
        && solve_logs.step_count % solver_config.particle_reordering_step_freq == 0) {
        logger::info_ln("SPH", "Reordering particles at step ", solve_logs.step_count);
        modules::ParticleReordering<Tvec, u_morton, Kern>(context, solver_config, storage)
            .reorder_particles();
    }

    sph_prestep(t_current, dt);

    using RTree = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    // Here we will add self grav to the external forces indep of vel (this will be moved into a
    // sperate module later)
    {

        auto constant_G = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");

        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>> set_constant_G(
            [&](shamrock::solvergraph::IDataEdge<Tscal> &constant_G) {
                constant_G.data = solver_config.get_constant_G();
            });

        set_constant_G.set_edges(constant_G);

        auto field_xyz = shamrock::solvergraph::FieldRefs<Tvec>::make_shared("", "");

        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::FieldRefs<Tvec>> set_field_xyz(
            [&](shamrock::solvergraph::FieldRefs<Tvec> &field_xyz_edge) {
                shamrock::solvergraph::DDPatchDataFieldRef<Tvec> field_xyz_refs = {};
                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    auto &field = pdat.get_field<Tvec>(ixyz);
                    field_xyz_refs.add_obj(p.id_patch, std::ref(field));
                });
                field_xyz_edge.set_refs(field_xyz_refs);
            });
        set_field_xyz.set_edges(field_xyz);

        const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");

        auto field_axyz_ext = shamrock::solvergraph::FieldRefs<Tvec>::make_shared("", "");

        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::FieldRefs<Tvec>>
            set_field_axyz_ext([&](shamrock::solvergraph::FieldRefs<Tvec> &field_axyz_ext_edge) {
                shamrock::solvergraph::DDPatchDataFieldRef<Tvec> field_axyz_ext_refs = {};
                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    auto &field = pdat.get_field<Tvec>(iaxyz_ext);
                    field_axyz_ext_refs.add_obj(p.id_patch, std::ref(field));
                });
                field_axyz_ext_edge.set_refs(field_axyz_ext_refs);
            });
        set_field_axyz_ext.set_edges(field_axyz_ext);

        auto sizes = shamrock::solvergraph::Indexes<u32>::make_shared("", "");

        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::Indexes<u32>> set_sizes(
            [&](shamrock::solvergraph::Indexes<u32> &sizes) {
                sizes.indexes = {};
                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    sizes.indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
                });
            });
        set_sizes.set_edges(sizes);

        auto gpart_mass = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");

        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>> set_gpart_mass(
            [&](shamrock::solvergraph::IDataEdge<Tscal> &gpart_mass) {
                gpart_mass.data = solver_config.gpart_mass;
            });

        set_gpart_mass.set_edges(gpart_mass);

        set_gpart_mass.evaluate();
        set_constant_G.evaluate();
        set_field_xyz.evaluate();
        set_field_axyz_ext.evaluate();
        set_sizes.evaluate();

        Tscal eps_grav = shambase::get_check_ref(
                             std::get_if<SelfGravConfig::SofteningPlummer>(
                                 &solver_config.self_grav_config.softening_mode))
                             .epsilon;

        if (solver_config.self_grav_config.is_none()) {
            // do nothing
        } else if (solver_config.self_grav_config.is_direct()) {

            SelfGravConfig::Direct &direct_config = shambase::get_check_ref(
                std::get_if<SelfGravConfig::Direct>(&solver_config.self_grav_config.config));

            modules::SGDirectPlummer<Tvec> self_gravity_direct_node(
                eps_grav, direct_config.reference_mode);
            self_gravity_direct_node.set_edges(
                sizes, gpart_mass, constant_G, field_xyz, field_axyz_ext);
            self_gravity_direct_node.evaluate();

        } else if (solver_config.self_grav_config.is_mm()) {

            SelfGravConfig::MM &mm_config = shambase::get_check_ref(
                std::get_if<SelfGravConfig::MM>(&solver_config.self_grav_config.config));

            auto run_sg_mm = [&](auto mm_order_tag) {
                constexpr u32 order = decltype(mm_order_tag)::value;
                modules::SGMMPlummer<Tvec, order> self_gravity_mm_node(
                    eps_grav, mm_config.opening_angle, mm_config.reduction_level);
                self_gravity_mm_node.set_edges(
                    sizes, gpart_mass, constant_G, field_xyz, field_axyz_ext);
                self_gravity_mm_node.evaluate();
            };

            switch (mm_config.order) {
            case 1 : run_sg_mm(std::integral_constant<u32, 1>{}); break;
            case 2 : run_sg_mm(std::integral_constant<u32, 2>{}); break;
            case 3 : run_sg_mm(std::integral_constant<u32, 3>{}); break;
            case 4 : run_sg_mm(std::integral_constant<u32, 4>{}); break;
            case 5 : run_sg_mm(std::integral_constant<u32, 5>{}); break;
            default: shambase::throw_unimplemented();
            }

        } else if (solver_config.self_grav_config.is_fmm()) {

            SelfGravConfig::FMM &mm_config = shambase::get_check_ref(
                std::get_if<SelfGravConfig::FMM>(&solver_config.self_grav_config.config));

            auto run_sg_fmm = [&](auto mm_order_tag) {
                constexpr u32 order = decltype(mm_order_tag)::value;
                modules::SGFMMPlummer<Tvec, order> self_gravity_mm_node(
                    eps_grav, mm_config.opening_angle, mm_config.reduction_level);
                self_gravity_mm_node.set_edges(
                    sizes, gpart_mass, constant_G, field_xyz, field_axyz_ext);
                self_gravity_mm_node.evaluate();
            };

            switch (mm_config.order) {
            case 1 : run_sg_fmm(std::integral_constant<u32, 1>{}); break;
            case 2 : run_sg_fmm(std::integral_constant<u32, 2>{}); break;
            case 3 : run_sg_fmm(std::integral_constant<u32, 3>{}); break;
            case 4 : run_sg_fmm(std::integral_constant<u32, 4>{}); break;
            case 5 : run_sg_fmm(std::integral_constant<u32, 5>{}); break;
            default: shambase::throw_unimplemented();
            }

        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Self gravity config not supported, current state is : \n"
                + nlohmann::json{solver_config.self_grav_config}.dump(4));
        }
    }

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
    auto &merged_xyzh                             = storage.merged_xyzh.get();
    shambase::DistributedData<RTree> &trees       = storage.merged_pos_trees.get();
    // ComputeField<Tscal> &omega                    = storage.omega.get();

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf      = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf       = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf       = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf      = ghost_layout.get_field_idx<Tscal>("omega");
    u32 iB_on_rho_interf   = (has_B_field) ? ghost_layout.get_field_idx<Tvec>("B/rho") : 0;
    u32 ipsi_on_rho_interf = (has_psi_field) ? ghost_layout.get_field_idx<Tscal>("psi/ch") : 0;

    using RTreeField = RadixTreeField<Tscal>;
    shambase::DistributedData<RTreeField> rtree_field_h;

    Tscal next_cfl = 0;

    u32 corrector_iter_cnt    = 0;
    bool need_rerun_corrector = false;
    do {

        reset_merge_ghosts_fields();
        reset_eos_fields();

        if (corrector_iter_cnt == 50) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "the corrector has made over 50 loops, either their is a bug, either you are using "
                "a dt that is too large");
        }

        // communicate fields
        communicate_merge_ghosts_fields();

        if (solver_config.has_field_alphaAV()) {

            std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> patchdatas
                = std::make_shared<shamrock::solvergraph::PatchDataLayerRefs>(
                    "patchdata_layer_ref", "patchdata_layer_ref");

            auto node_set_edge = scheduler().get_node_set_edge_patchdata_layer_refs();
            node_set_edge->set_edges(patchdatas);
            node_set_edge->evaluate();

            shamrock::solvergraph::CopyPatchDataFieldFromLayer<Tscal> node_copy(
                scheduler().get_layout_ptr(), "alpha_AV");
            node_copy.set_edges(patchdatas, storage.alpha_av_updated);
            node_copy.evaluate();
        }

        if (solver_config.has_field_dtdivv()) {

            if (solver_config.combined_dtdiv_divcurlv_compute) {
                if (solver_config.has_field_dtdivv()) {
                    sph::modules::DiffOperatorDtDivv<Tvec, Kern>(context, solver_config, storage)
                        .update_dtdivv(true);
                }
            } else {

                if (solver_config.has_field_divv()) {
                    sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
                        .update_divv();
                }

                if (solver_config.has_field_curlv()) {
                    sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
                        .update_curlv();
                }

                if (solver_config.has_field_dtdivv()) {
                    sph::modules::DiffOperatorDtDivv<Tvec, Kern>(context, solver_config, storage)
                        .update_dtdivv(false);
                }
            }

        } else {
            if (solver_config.has_field_divv()) {
                sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
                    .update_divv();
            }

            if (solver_config.has_field_curlv()) {
                sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
                    .update_curlv();
            }
        }

        // if (solver_config.has_field_divB()) {
        //     sph::modules::DiffOperatorsB<Tvec, Kern>(context, solver_config, storage)
        //         .update_divB();
        // }

        // if (solver_config.has_field_curlB()) {
        //     sph::modules::DiffOperatorsB<Tvec, Kern>(context, solver_config, storage)
        //         .update_curlB();
        // }
        update_artificial_viscosity(dt);

        if (solver_config.has_field_alphaAV()) {

            shamrock::solvergraph::Field<Tscal> &comp_field_send
                = shambase::get_check_ref(storage.alpha_av_updated);

            using InterfaceBuildInfos =
                typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

            shambase::Timer time_interf;
            time_interf.start();

            auto field_interf = ghost_handle.template build_interface_native<PatchDataField<Tscal>>(
                storage.ghost_patch_cache.get(),
                [&](u64 sender,
                    u64 /*receiver*/,
                    InterfaceBuildInfos binfo,
                    sham::DeviceBuffer<u32> &buf_idx,
                    u32 cnt) -> PatchDataField<Tscal> {
                    PatchDataField<Tscal> &sender_field = comp_field_send.get_field(sender);

                    return sender_field.make_new_from_subset(buf_idx, cnt);
                });

            shambase::DistributedDataShared<PatchDataField<Tscal>> interf_pdat
                = ghost_handle.communicate_pdatfield(std::move(field_interf), 1);

            shambase::DistributedData<PatchDataField<Tscal>> merged_field
                = ghost_handle.template merge_native<PatchDataField<Tscal>, PatchDataField<Tscal>>(
                    std::move(interf_pdat),
                    [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                        PatchDataField<Tscal> &receiver_field
                            = comp_field_send.get_field(p.id_patch);
                        return receiver_field.duplicate();
                    },
                    [](PatchDataField<Tscal> &mpdat, PatchDataField<Tscal> &pdat_interf) {
                        mpdat.insert(pdat_interf);
                    });

            time_interf.end();
            storage.timings_details.interface += time_interf.elasped_sec();

            storage.alpha_av_ghost.set(std::move(merged_field));
        }

        // compute pressure
        compute_eos_fields();

        constexpr bool debug_interfaces = false;
        if constexpr (debug_interfaces) {

            if (solver_config.do_debug_dump) {

                shambase::DistributedData<MergedPatchData> &mpdat
                    = storage.merged_patchdata_ghost.get();

                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                    MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
                    PatchDataLayer &mpdat         = merged_patch.pdat;

                    sycl::buffer<Tvec> &buf_xyz = shambase::get_check_ref(
                        merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
                    sycl::buffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
                    sycl::buffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);

                    u32 total_elements = shambase::get_check_ref(storage.part_counts_with_ghost)
                                             .indexes.get(cur_p.id_patch);
                    SHAM_ASSERT(merged_patch.total_elements == total_elements);

                    Debug_ph_dump<Tvec> info{
                        total_elements,
                        solver_config.gpart_mass,

                        buf_xyz,
                        buf_hpart,
                        buf_vxyz};

                    make_interface_debug_phantom_dump(info).gen_file().write_to_file(
                        solver_config.debug_dump_filename);
                    logger::raw_ln("writing : ", solver_config.debug_dump_filename);
                });
            }
        }

        // compute force
        shamlog_debug_ln("sph::BasicGas", "compute force");

        // save old acceleration
        prepare_corrector();

        update_derivs();

        modules::ConservativeCheck<Tvec, Kern> cv_check(context, solver_config, storage);
        cv_check.check_conservation();

        ComputeField<Tscal> vepsilon_v_sq
            = utility.make_compute_field<Tscal>("vmean epsilon_v^2", 1);
        ComputeField<Tscal> uepsilon_u_sq
            = utility.make_compute_field<Tscal>("umean epsilon_u^2", 1);

        // corrector
        shamlog_debug_ln("sph::BasicGas", "leapfrog corrector");
        utility.fields_leapfrog_corrector<Tvec>(
            ivxyz, iaxyz, storage.old_axyz.get(), vepsilon_v_sq, dt / 2);
        utility.fields_leapfrog_corrector<Tscal>(
            iuint, iduint, storage.old_duint.get(), uepsilon_u_sq, dt / 2);

        if (solver_config.has_field_B_on_rho()) {
            ComputeField<Tscal> BOR_epsilon_BOR_sq
                = utility.make_compute_field<Tscal>("B/rho epsilon_B/rho^2", 1);
            utility.fields_leapfrog_corrector<Tvec>(
                iB_on_rho, idB_on_rho, storage.old_dB_on_rho.get(), BOR_epsilon_BOR_sq, dt / 2);
        }
        if (solver_config.has_field_B_on_rho()) {
            ComputeField<Tscal> POC_epsilon_POC_sq
                = utility.make_compute_field<Tscal>("psi/ch epsilon_psi/ch^2", 1);
            utility.fields_leapfrog_corrector<Tscal>(
                ipsi_on_ch, idpsi_on_ch, storage.old_dpsi_on_ch.get(), POC_epsilon_POC_sq, dt / 2);
        }

        if (solver_config.dust_config.has_epsilon_field()) {
            ComputeField<Tscal> epsilon_epsilon_sq
                = utility.make_compute_field<Tscal>("epsilon epsilon^2", 1);
            utility.fields_leapfrog_corrector<Tscal>(
                iepsilon, idtepsilon, storage.old_dtepsilon.get(), epsilon_epsilon_sq, dt / 2);
        }

        if (solver_config.dust_config.has_deltav_field()) {
            ComputeField<Tscal> epsilon_deltav_sq
                = utility.make_compute_field<Tscal>("deltav deltav^2", 1);
            utility.fields_leapfrog_corrector<Tvec>(
                ideltav, idtdeltav, storage.old_dtdeltav.get(), epsilon_deltav_sq, dt / 2);
        }

        storage.old_axyz.reset();
        storage.old_duint.reset();
        if (solver_config.has_field_B_on_rho()) {
            storage.old_dB_on_rho.reset();
        }
        if (solver_config.has_field_B_on_rho()) {
            storage.old_dpsi_on_ch.reset();
        }

        if (solver_config.dust_config.has_epsilon_field()) {
            storage.old_dtepsilon.reset();
        }

        if (solver_config.dust_config.has_deltav_field()) {
            storage.old_dtdeltav.reset();
        }

        Tscal rank_veps_v = sycl::sqrt(vepsilon_v_sq.compute_rank_max());
        ///////////////////////////////////////////
        // compute means //////////////////////////
        ///////////////////////////////////////////

        Tscal sum_vsq = utility.compute_rank_dot_sum<Tvec>(ivxyz);

        Tscal vmean_sq = shamalgs::collective::allreduce_sum(sum_vsq) / Tscal(Npart_all);

        Tscal vmean = sycl::sqrt(vmean_sq);

        Tscal rank_eps_v = rank_veps_v / vmean;

        if (vmean <= 0) {
            rank_eps_v = 0;
        }

        Tscal eps_v = shamalgs::collective::allreduce_max(rank_eps_v);

        shamlog_debug_ln("BasicGas", "epsilon v :", eps_v);

        if (eps_v > 1e-2) {
            if (shamcomm::world_rank() == 0) {
                logger::warn_ln(
                    "BasicGasSPH",
                    shambase::format(
                        "the corrector tolerance are broken the step will "
                        "be re rerunned\n    eps_v = {}",
                        eps_v));
            }
            need_rerun_corrector = true;
            solver_config.time_state.cfl_multiplier /= 2;

            // logger::info_ln("rerun corrector ...");
        } else {
            need_rerun_corrector = false;
        }

        if (!need_rerun_corrector) {

            sink_update.corrector_step(dt);

            // write back alpha av field
            if (solver_config.has_field_alphaAV()) {

                const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");
                shamrock::solvergraph::Field<Tscal> &alpha_av_updated
                    = shambase::get_check_ref(storage.alpha_av_updated);

                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                    sham::DeviceBuffer<Tscal> &buf_alpha_av
                        = pdat.get_field<Tscal>(ialpha_AV).get_buf();
                    sham::DeviceBuffer<Tscal> &buf_alpha_av_updated
                        = alpha_av_updated.get_field(cur_p.id_patch).get_buf();

                    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
                    sham::EventList depends_list;

                    auto alpha_av         = buf_alpha_av.get_write_access(depends_list);
                    auto alpha_av_updated = buf_alpha_av_updated.get_read_access(depends_list);

                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        shambase::parallel_for(
                            cgh, pdat.get_obj_cnt(), "write back alpha_av", [=](i32 id_a) {
                                alpha_av[id_a] = alpha_av_updated[id_a];
                            });
                    });

                    buf_alpha_av.complete_event_state(e);
                    buf_alpha_av_updated.complete_event_state(e);
                });
            }

            shamlog_debug_ln("BasicGas", "computing next CFL");

            ComputeField<Tscal> vsig_max_dt = utility.make_compute_field<Tscal>("vsig_a", 1);
            std::unique_ptr<ComputeField<Tscal>> vclean_dt;
            if (has_psi_field) {
                vclean_dt = std::make_unique<ComputeField<Tscal>>(
                    utility.make_compute_field<Tscal>("vclean_a", 1));
            }

            shambase::DistributedData<PatchDataLayer> &mpdats
                = storage.merged_patchdata_ghost.get();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

                sham::DeviceBuffer<Tvec> &buf_xyz
                    = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
                sham::DeviceBuffer<Tvec> &buf_vxyz = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
                sham::DeviceBuffer<Tscal> &buf_hpart
                    = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
                sham::DeviceBuffer<Tscal> &buf_uint = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
                sham::DeviceBuffer<Tscal> &buf_pressure
                    = shambase::get_check_ref(storage.pressure).get_field(cur_p.id_patch).get_buf();
                sham::DeviceBuffer<Tscal> &cs_buf = shambase::get_check_ref(storage.soundspeed)
                                                        .get_field(cur_p.id_patch)
                                                        .get_buf();

                sham::DeviceBuffer<Tscal> &vsig_buf = vsig_max_dt.get_buf_check(cur_p.id_patch);

                sycl::range range_npart{pdat.get_obj_cnt()};

                tree::ObjectCache &pcache
                    = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

                /////////////////////////////////////////////

                {

                    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
                    sham::EventList depends_list;

                    auto xyz                  = buf_xyz.get_read_access(depends_list);
                    auto vxyz                 = buf_vxyz.get_read_access(depends_list);
                    auto hpart                = buf_hpart.get_read_access(depends_list);
                    auto u                    = buf_uint.get_read_access(depends_list);
                    auto pressure             = buf_pressure.get_read_access(depends_list);
                    auto cs                   = cs_buf.get_read_access(depends_list);
                    auto vsig                 = vsig_buf.get_write_access(depends_list);
                    auto particle_looper_ptrs = pcache.get_read_access(depends_list);

                    NamedStackEntry tmppp{"compute vsig"};
                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        const Tscal pmass    = solver_config.gpart_mass;
                        const Tscal alpha_u  = 1.0;
                        const Tscal alpha_AV = 1.0;
                        const Tscal beta_AV  = 2.0;

                        tree::ObjectCacheIterator particle_looper(particle_looper_ptrs);

                        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                        shambase::parallel_for(
                            cgh, pdat.get_obj_cnt(), "compute vsig", [=](i32 id_a) {
                                using namespace shamrock::sph;

                                Tvec sum_axyz  = {0, 0, 0};
                                Tscal sum_du_a = 0;
                                Tscal h_a      = hpart[id_a];

                                Tvec xyz_a  = xyz[id_a];
                                Tvec vxyz_a = vxyz[id_a];

                                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                                Tscal rho_a_sq  = rho_a * rho_a;
                                Tscal rho_a_inv = 1. / rho_a;

                                Tscal P_a = pressure[id_a];

                                const Tscal u_a = u[id_a];

                                Tscal cs_a = cs[id_a];

                                Tscal vsig_max = 0;

                                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                                    // compute only omega_a
                                    Tvec dr    = xyz_a - xyz[id_b];
                                    Tscal rab2 = sycl::dot(dr, dr);
                                    Tscal h_b  = hpart[id_b];

                                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                                        return;
                                    }

                                    Tscal rab       = sycl::sqrt(rab2);
                                    Tvec vxyz_b     = vxyz[id_b];
                                    Tvec v_ab       = vxyz_a - vxyz_b;
                                    const Tscal u_b = u[id_b];

                                    Tvec r_ab_unit = dr / rab;

                                    if (rab < 1e-9) {
                                        r_ab_unit = {0, 0, 0};
                                    }

                                    Tscal rho_b         = rho_h(pmass, h_b, Kernel::hfactd);
                                    Tscal P_b           = pressure[id_b];
                                    Tscal cs_b          = cs[id_b];
                                    Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
                                    Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                                    /////////////////
                                    // internal energy update
                                    //  scalar : f32  | vector : f32_3
                                    const Tscal alpha_a = alpha_AV;
                                    const Tscal alpha_b = alpha_AV;

                                    Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;

                                    vsig_max = sycl::fmax(vsig_max, vsig_a);
                                });

                                vsig[id_a] = vsig_max;
                            });
                    });

                    if (has_psi_field) {
                        NamedStackEntry tmppp{"compute vclean"};
                        Tscal const mu_0 = solver_config.get_constant_mu_0();
                        sham::DeviceBuffer<Tscal> &vclean_buf
                            = shambase::get_check_ref(vclean_dt).get_buf_check(cur_p.id_patch);

                        Tvec *B_on_rho = mpdat.get_field_buf_ref<Tvec>(iB_on_rho_interf)
                                             .get_write_access(depends_list);

                        auto vclean = vclean_buf.get_write_access(depends_list);

                        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                            const Tscal pmass = solver_config.gpart_mass;

                            tree::ObjectCacheIterator particle_looper(particle_looper_ptrs);

                            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                            shambase::parallel_for(
                                cgh, pdat.get_obj_cnt(), "compute vclean", [=](i32 id_a) {
                                    using namespace shamrock::sph;

                                    Tscal h_a       = hpart[id_a];
                                    Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                                    const Tscal u_a = u[id_a];
                                    Tscal cs_a      = cs[id_a];
                                    Tvec B_a        = B_on_rho[id_a] * rho_a;

                                    Tscal vclean_a = shamphys::MHD_physics<Tvec, Tscal>::v_shock(
                                        cs_a, B_a, rho_a, mu_0);

                                    vclean[id_a] = vclean_a;
                                });
                        });
                        mpdat.get_field_buf_ref<Tvec>(iB_on_rho_interf).complete_event_state(e);
                        vclean_buf.complete_event_state(e);
                    };

                    buf_xyz.complete_event_state(e);
                    buf_vxyz.complete_event_state(e);
                    buf_hpart.complete_event_state(e);
                    buf_uint.complete_event_state(e);
                    buf_pressure.complete_event_state(e);
                    cs_buf.complete_event_state(e);
                    vsig_buf.complete_event_state(e);

                    sham::EventList resulting_events;
                    resulting_events.add_event(e);
                    pcache.complete_event_state(resulting_events);
                }
            });

            ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", 1);

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

                sham::DeviceBuffer<Tvec> &buf_axyz = pdat.get_field<Tvec>(iaxyz).get_buf();
                sham::DeviceBuffer<Tscal> &buf_hpart
                    = mpdat.get_field<Tscal>(ihpart_interf).get_buf();
                sham::DeviceBuffer<Tscal> &vsig_buf   = vsig_max_dt.get_buf_check(cur_p.id_patch);
                sham::DeviceBuffer<Tscal> &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

                auto &q = shamsys::instance::get_compute_scheduler().get_queue();
                sham::EventList depends_list;

                auto hpart  = buf_hpart.get_read_access(depends_list);
                auto a      = buf_axyz.get_read_access(depends_list);
                auto vsig   = vsig_buf.get_read_access(depends_list);
                auto cfl_dt = cfl_dt_buf.get_write_access(depends_list);

                auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                    Tscal C_cour = solver_config.cfl_config.cfl_cour
                                   * solver_config.time_state.cfl_multiplier;
                    Tscal C_force = solver_config.cfl_config.cfl_force
                                    * solver_config.time_state.cfl_multiplier;

                    cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                        Tscal h_a     = hpart[item];
                        Tscal vsig_a  = vsig[item];
                        Tscal abs_a_a = sycl::length(a[item]);

                        Tscal dt_c = C_cour * h_a / vsig_a;
                        Tscal dt_f = C_force * sycl::sqrt(h_a / abs_a_a);

                        cfl_dt[item] = sycl::min(dt_c, dt_f);
                    });
                });

                if (has_psi_field) {
                    sham::DeviceBuffer<Tscal> &vclean_buf
                        = shambase::get_check_ref(vclean_dt).get_buf_check(cur_p.id_patch);
                    auto vclean = vclean_buf.get_read_access(depends_list);
                    auto e      = q.submit(depends_list, [&](sycl::handler &cgh) {
                        Tscal C_cour = solver_config.cfl_config.cfl_cour
                                       * solver_config.time_state.cfl_multiplier;

                        cgh.parallel_for(
                            sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                                Tscal h_a      = hpart[item];
                                Tscal vclean_a = vclean[item];

                                Tscal dt_divB_cleaning = C_cour * h_a / vclean_a;

                                cfl_dt[item] = sycl::min(cfl_dt[item], dt_divB_cleaning);
                            });
                    });
                    vclean_buf.complete_event_state(e);
                };

                buf_hpart.complete_event_state(e);
                buf_axyz.complete_event_state(e);
                vsig_buf.complete_event_state(e);
                cfl_dt_buf.complete_event_state(e);
            });

            Tscal rank_dt = cfl_dt.compute_rank_min();

            Tscal sink_sink_cfl = shambase::get_infty<Tscal>();
            if (!storage.sinks.is_empty()) {
                // sink sink CFL

                Tscal G = solver_config.get_constant_G();

                Tscal C_force
                    = solver_config.cfl_config.cfl_force * solver_config.time_state.cfl_multiplier;
                Tscal eta_phi = solver_config.cfl_config.eta_sink;

                std::vector<SinkParticle<Tvec>> &sink_parts = storage.sinks.get();

                for (u32 i = 0; i < sink_parts.size(); i++) {
                    SinkParticle<Tvec> &s_i = sink_parts[i];
                    Tscal sink_sink_cfl_i   = shambase::get_infty<Tscal>();

                    Tvec f_i = s_i.ext_acceleration;

                    Tscal grad_phi_i_sq = sham::dot(f_i, f_i); // m^2.s^-4

                    if (grad_phi_i_sq == 0) {
                        continue;
                    }

                    for (u32 j = 0; j < sink_parts.size(); j++) {
                        SinkParticle<Tvec> &s_j = sink_parts[j];

                        if (i == j) {
                            continue;
                        }

                        Tvec rij       = s_i.pos - s_j.pos;
                        Tscal rij_scal = sycl::length(rij);

                        Tscal phi_ij  = G * s_j.mass / rij_scal;           // J / kg = m^2.s^-2
                        Tscal term_ij = sham::abs(phi_ij) / grad_phi_i_sq; // s^2
                        Tscal dt_ij   = C_force * eta_phi * sycl::sqrt(term_ij); // s

                        sink_sink_cfl_i = sham::min(sink_sink_cfl_i, dt_ij);
                    }

                    sink_sink_cfl = sham::min(sink_sink_cfl, sink_sink_cfl_i);
                }

                sink_sink_cfl = shamalgs::collective::allreduce_min(sink_sink_cfl);
            }

            shamlog_debug_ln("BasigGas", "rank", shamcomm::world_rank(), "found cfl dt =", rank_dt);

            Tscal hydro_cfl = shamalgs::collective::allreduce_min(rank_dt);

            if (shamcomm::world_rank() == 0) {
                shamlog_info_ln("SPH", "CFL hydro =", hydro_cfl, "sink sink =", sink_sink_cfl);
            }

            next_cfl = sham::min(hydro_cfl, sink_sink_cfl);

            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "sph::Model",
                    "cfl dt =",
                    next_cfl,
                    "cfl multiplier :",
                    solver_config.time_state.cfl_multiplier);
            }

            // this should not be needed idealy, but we need the pressure on the ghosts and
            // we don't want to communicate it as it can be recomputed from the other fields
            // hence we copy the soundspeed at the end of the step to a field in the patchdata
            if (solver_config.has_field_soundspeed()) {

                const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");

                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                    sham::DeviceBuffer<Tscal> &buf_cs = pdat.get_field_buf_ref<Tscal>(isoundspeed);
                    sham::DeviceBuffer<Tscal> &buf_cs_in
                        = shambase::get_check_ref(storage.soundspeed)
                              .get_field(cur_p.id_patch)
                              .get_buf();

                    sycl::range range_npart{pdat.get_obj_cnt()};

                    /////////////////////////////////////////////

                    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
                    sham::EventList depends_list;

                    auto cs_in = buf_cs_in.get_read_access(depends_list);
                    auto cs    = buf_cs.get_write_access(depends_list);

                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        const Tscal pmass = solver_config.gpart_mass;

                        cgh.parallel_for(
                            sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                                cs[item] = cs_in[item];
                            });
                    });

                    buf_cs_in.complete_event_state(e);
                    buf_cs.complete_event_state(e);
                });
            }

        } // if (!need_rerun_corrector) {

        corrector_iter_cnt++;

        if (solver_config.has_field_alphaAV()) {
            storage.alpha_av_ghost.reset();
        }

    } while (need_rerun_corrector);

    reset_merge_ghosts_fields();
    reset_eos_fields();

    // if delta too big jump to compute force

    tstep.end();

    sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

    f64 delta_mpi_timer = shamcomm::mpi::get_timer("total") - mpi_timer_start;
    f64 t_dev_alloc
        = (mem_perf_infos_end.time_alloc_device - mem_perf_infos_start.time_alloc_device)
          + (mem_perf_infos_end.time_free_device - mem_perf_infos_start.time_free_device);

    u64 rank_count = scheduler().get_rank_count();
    f64 rate       = f64(rank_count) / tstep.elasped_sec();

    u64 npatch = scheduler().patch_list.local.size();

    // logger::info_ln("SPHSolver", "process rate : ", rate, "particle.s-1");

    std::string log_step = report_perf_timestep(
        rate,
        rank_count,
        npatch,
        tstep.elasped_sec(),
        delta_mpi_timer,
        t_dev_alloc,
        mem_perf_infos_end.max_allocated_byte_device);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sph::Model", log_step);
        logger::info_ln(
            "sph::Model", "estimated rate :", dt * (3600 / tstep.elasped_sec()), "(tsim/hr)");
    }

    solve_logs.register_log(
        {t_current,              // f64 solver_t;
         dt,                     // f64 solver_dt;
         shamcomm::world_rank(), // i32 world_rank;
         rank_count,             // u64 rank_count;
         rate,                   // f64 rate;
         tstep.elasped_sec(),    // f64 elasped_sec;
         shambase::details::get_wtime()});

    storage.timings_details.reset();

    reset_serial_patch_tree();
    reset_ghost_handler();

    shambase::get_check_ref(storage.part_counts).free_alloc();
    shambase::get_check_ref(storage.part_counts_with_ghost).free_alloc();
    shambase::get_check_ref(storage.positions_with_ghosts).free_alloc();
    shambase::get_check_ref(storage.hpart_with_ghosts).free_alloc();
    storage.merged_xyzh.reset();
    shambase::get_check_ref(storage.omega).free_alloc();
    clear_merged_pos_trees();
    clear_ghost_cache();
    reset_presteps_rint();
    reset_neighbors_cache();

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    solver_config.set_next_dt(next_cfl);
    solver_config.set_time(t_current + dt);

    auto get_next_cfl_mult = [&]() {
        Tscal cfl_m = solver_config.time_state.cfl_multiplier;
        Tscal stiff = solver_config.cfl_config.cfl_multiplier_stiffness;

        return (cfl_m * stiff + 1.) / (stiff + 1.);
    };

    solver_config.time_state.cfl_multiplier = get_next_cfl_mult();

    TimestepLog log;
    log.rank     = shamcomm::world_rank();
    log.rate     = rate;
    log.npart    = rank_count;
    log.tcompute = tstep.elasped_sec();

    return log;
}

using namespace shammath;

template class shammodels::sph::Solver<f64_3, M4>;
template class shammodels::sph::Solver<f64_3, M6>;
template class shammodels::sph::Solver<f64_3, M8>;

template class shammodels::sph::Solver<f64_3, C2>;
template class shammodels::sph::Solver<f64_3, C4>;
template class shammodels::sph::Solver<f64_3, C6>;
