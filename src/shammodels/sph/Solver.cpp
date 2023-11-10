// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "Solver.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/reduction.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamcomm/collectives.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SPHSolverImpl.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/modules/ComputeEos.hpp"
#include "shammodels/sph/modules/ConservativeCheck.hpp"
#include "shammodels/sph/modules/DiffOperator.hpp"
#include "shammodels/sph/modules/DiffOperatorDtDivv.hpp"
#include "shammodels/sph/modules/ExternalForces.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shammodels/sph/modules/UpdateViscosity.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamrock/tree/TreeTraversalCache.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

template<class Tvec, template<class> class Kern>
using SPHSolve = shammodels::sph::Solver<Tvec, Kern>;

template<class vec>
shamrock::LegacyVtkWritter start_dump(PatchScheduler &sched, std::string dump_name) {
    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

    using namespace shamrock::patch;

    u64 num_obj = sched.get_rank_count();

    logger::debug_mpi_ln("sph::BasicGas", "rank count =", num_obj);

    std::unique_ptr<sycl::buffer<vec>> pos = sched.rankgather_field<vec>(0);

    writer.write_points(pos, num_obj);

    return writer;
}

void vtk_dump_add_patch_id(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
    StackEntry stack_loc{};

    u64 num_obj = sched.get_rank_count();

    using namespace shamrock::patch;

    if (num_obj > 0) {
        // TODO aggregate field ?
        sycl::buffer<u64> idp(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(idp, cur_p.id_patch, ptr, pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field("patchid", idp, num_obj);
    } else {
        writter.write_field_no_buf<u64>("patchid");
    }
}

void vtk_dump_add_worldrank(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    u64 num_obj = sched.get_rank_count();

    if (num_obj > 0) {

        // TODO aggregate field ?
        sycl::buffer<u32> idp(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into<u32>(idp, shamcomm::world_rank(), ptr, pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field("world_rank", idp, num_obj);

    } else {
        writter.write_field_no_buf<u32>("world_rank");
    }
}

template<class T>
void vtk_dump_add_compute_field(
    PatchScheduler &sched,
    shamrock::LegacyVtkWritter &writter,
    shamrock::ComputeField<T> &field,
    std::string field_dump_name) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<T>> field_vals = field.rankgather_computefield(sched);

    writter.write_field(field_dump_name, field_vals, num_obj);
}

template<class T>
void vtk_dump_add_field(
    PatchScheduler &sched,
    shamrock::LegacyVtkWritter &writter,
    u32 field_idx,
    std::string field_dump_name) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<T>> field_vals = sched.rankgather_field<T>(field_idx);

    writter.write_field(field_dump_name, field_vals, num_obj);
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::gen_serial_patch_tree() {
    StackEntry stack_loc{};

    SerialPatchTree<Tvec> _sptree = SerialPatchTree<Tvec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::apply_position_boundary(Tscal time_val) {

    StackEntry stack_loc{};

    logger::debug_ln("SphSolver", "apply position boundary");

    PatchScheduler &sched = scheduler();

    shamrock::SchedulerUtility integrators(sched);
    shamrock::ReattributeDataUtility reatrib(sched);

    const u32 ixyz    = sched.pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz   = sched.pdl.get_field_idx<Tvec>("vxyz");
    auto [bmin, bmax] = sched.get_box_volume<Tvec>();

    using SolverConfigBC           = typename Config::BCConfig;
    using SolverBCFree             = typename SolverConfigBC::Free;
    using SolverBCPeriodic         = typename SolverConfigBC::Periodic;
    using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;
    if (SolverBCFree *c = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
        logger::info_ln("PositionUpdated", "free boundaries skipping geometry update");
    } else if (
        SolverBCPeriodic *c =
            std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
        integrators.fields_apply_periodicity(ixyz, std::pair{bmin, bmax});
    } else if (
        SolverBCShearingPeriodic *c =
            std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)) {
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
void SPHSolve<Tvec, Kern>::build_ghost_cache() {

    StackEntry stack_loc{};

    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;
    SPHUtils sph_utils(scheduler());

    storage.ghost_patch_cache.set(sph_utils.build_interf_cache(
        storage.ghost_handler.get(), storage.serial_patch_tree.get(), htol_up_tol));
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::clear_ghost_cache() {
    StackEntry stack_loc{};
    storage.ghost_patch_cache.reset();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::merge_position_ghost() {

    StackEntry stack_loc{};

    storage.merged_xyzh.set(
        storage.ghost_handler.get().build_comm_merge_positions(storage.ghost_patch_cache.get()));
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::build_merged_pos_trees() {

    StackEntry stack_loc{};

    SPHSolverImpl solver(context);

    constexpr u32 reduc_level = 3;
    auto &merged_xyzh         = storage.merged_xyzh.get();
    storage.merged_pos_trees.set(solver.make_merge_patch_trees(merged_xyzh, reduc_level));
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::clear_merged_pos_trees() {
    StackEntry stack_loc{};
    storage.merged_pos_trees.reset();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::do_predictor_leapfrog(Tscal dt) {

    StackEntry stack_loc{};
    using namespace shamrock::patch;
    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz       = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint      = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint     = pdl.get_field_idx<Tscal>("duint");

    shamrock::SchedulerUtility utility(scheduler());

    // forward euler step f dt/2
    logger::debug_ln("sph::BasicGas", "forward euler step f dt/2");
    utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
    utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);

    // forward euler step positions dt
    logger::debug_ln("sph::BasicGas", "forward euler step positions dt");
    utility.fields_forward_euler<Tvec>(ixyz, ivxyz, dt);

    // forward euler step f dt/2
    logger::debug_ln("sph::BasicGas", "forward euler step f dt/2");
    utility.fields_forward_euler<Tvec>(ivxyz, iaxyz, dt / 2);
    utility.fields_forward_euler<Tscal>(iuint, iduint, dt / 2);
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::sph_prestep(Tscal time_val) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    using RTree    = RadixTree<u_morton, Tvec>;
    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;

    SPHUtils sph_utils(scheduler());
    shamrock::SchedulerUtility utility(scheduler());
    SPHSolverImpl solver(context);

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ihpart     = pdl.get_field_idx<Tscal>("hpart");

    ComputeField<Tscal> _epsilon_h, _h_old;

    u32 hstep_cnt = 0;
    for (; hstep_cnt < 100; hstep_cnt++) {

        gen_ghost_handler(time_val);
        build_ghost_cache();
        merge_position_ghost();
        build_merged_pos_trees();
        compute_presteps_rint();
        start_neighbors_cache();

        _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1, Tscal(100));
        _h_old     = utility.save_field<Tscal>(ihpart, "h_old");

        Tscal max_eps_h;

        u32 iter_h = 0;
        for (; iter_h < 50; iter_h++) {
            NamedStackEntry stack_loc2{"iterate smoothing length"};
            // iterate smoothing length
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
                logger::debug_ln("SPHLeapfrog", "patch : n°", p.id_patch, "->", "h iteration");

                sycl::buffer<Tscal> &eps_h =
                    shambase::get_check_ref(_epsilon_h.get_buf(p.id_patch));
                sycl::buffer<Tscal> &hold = shambase::get_check_ref(_h_old.get_buf(p.id_patch));

                sycl::buffer<Tscal> &hnew =
                    shambase::get_check_ref(pdat.get_field<Tscal>(ihpart).get_buf());
                sycl::buffer<Tvec> &merged_r = shambase::get_check_ref(
                    storage.merged_xyzh.get().get(p.id_patch).field_pos.get_buf());

                sycl::range range_npart{pdat.get_obj_cnt()};

                RTree &tree = storage.merged_pos_trees.get().get(p.id_patch);

                tree::ObjectCache &neigh_cache =
                    storage.neighbors_cache.get().get_cache(p.id_patch);

                sph_utils.iterate_smoothing_length_cache(
                    merged_r,
                    hnew,
                    hold,
                    eps_h,
                    range_npart,
                    neigh_cache,
                    solver_config.gpart_mass,
                    htol_up_tol,
                    htol_up_iter);
                // sph_utils.iterate_smoothing_length_tree(merged_r, hnew, hold, eps_h, range_npart,
                // tree, gpart_mass, htol_up_tol, htol_up_iter);
            });
            max_eps_h = _epsilon_h.compute_rank_max();
            if (max_eps_h < 1e-6) {
                logger::debug_sycl("Smoothinglength", "converged at i =", iter_h);
                break;
            }
        }

        // logger::info_ln("Smoothinglength", "eps max =", max_eps_h);

        Tscal min_eps_h = shamalgs::collective::allreduce_min(_epsilon_h.compute_rank_min());
        if (min_eps_h == -1) {
            if (shamcomm::world_rank() == 0) {
                logger::warn_ln(
                    "Smoothinglength",
                    "smoothing length is not converged, rerunning the iterator ...");
            }

            reset_ghost_handler();
            clear_ghost_cache();
            storage.merged_xyzh.reset();
            clear_merged_pos_trees();
            reset_presteps_rint();
            reset_neighbors_cache();

            continue;
        } else {
            if (shamcomm::world_rank() == 0) {

                std::string log = "";
                log += "smoothing length iteration converged\n";
                log += shambase::format(
                    "  eps min = {}, max = {}\n  iterations = {}", min_eps_h, max_eps_h, iter_h);

                logger::info_ln("Smoothinglength", log);
            }
        }

        //// compute omega
        storage.omega.set(utility.make_compute_field<Tscal>("omega", 1));
        {

            ComputeField<Tscal> &omega = storage.omega.get();
            NamedStackEntry stack_loc2{"compute omega"};

            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
                logger::debug_ln("SPHLeapfrog", "patch : n°", p.id_patch, "->", "h iteration");

                sycl::buffer<Tscal> &omega_h = shambase::get_check_ref(omega.get_buf(p.id_patch));

                sycl::buffer<Tscal> &hnew =
                    shambase::get_check_ref(pdat.get_field<Tscal>(ihpart).get_buf());
                sycl::buffer<Tvec> &merged_r = shambase::get_check_ref(
                    storage.merged_xyzh.get().get(p.id_patch).field_pos.get_buf());

                sycl::range range_npart{pdat.get_obj_cnt()};

                RTree &tree = storage.merged_pos_trees.get().get(p.id_patch);

                tree::ObjectCache &neigh_cache =
                    storage.neighbors_cache.get().get_cache(p.id_patch);
                ;

                sph_utils.compute_omega(
                    merged_r, hnew, omega_h, range_npart, neigh_cache, solver_config.gpart_mass);
            });
        }
        _epsilon_h.reset();
        _h_old.reset();
        break;
    }

    if (hstep_cnt == 100) {
        logger::err_ln("SPH", "the h iterator is not converged after", hstep_cnt, "iterations");
    }
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::init_ghost_layout() {

    storage.ghost_layout.set(shamrock::patch::PatchDataLayout{});

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

    ghost_layout.add_field<Tscal>("hpart", 1);
    ghost_layout.add_field<Tscal>("uint", 1);
    ghost_layout.add_field<Tvec>("vxyz", 1);
    ghost_layout.add_field<Tscal>("omega", 1);

    if(solver_config.ghost_has_soundspeed()){
        ghost_layout.add_field<Tscal>("soundspeed", 1);
    }

    if (solver_config.has_field_alphaAV()) {
        ghost_layout.add_field<Tscal>("alpha_AV", 1);
    }
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::compute_presteps_rint() {

    StackEntry stack_loc{};

    auto &xyzh_merged = storage.merged_xyzh.get();

    storage.rtree_rint_field.set(storage.merged_pos_trees.get().template map<RadixTreeField<Tscal>>(
        [&](u64 id, RTree &rtree) {
            PreStepMergedField &tmp = xyzh_merged.get(id);

            return rtree.compute_int_boxes(
                shamsys::instance::get_compute_queue(), tmp.field_hpart.get_buf(), htol_up_tol);
        }));
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::reset_presteps_rint() {
    storage.rtree_rint_field.reset();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::start_neighbors_cache() {

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    storage.neighbors_cache.set(shamrock::tree::ObjectCacheHandler(u64(10e9), [&](u64 patch_id) {
        logger::debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        PreStepMergedField &mfield = storage.merged_xyzh.get().get(patch_id);

        sycl::buffer<Tvec> &buf_xyz    = shambase::get_check_ref(mfield.field_pos.get_buf());
        sycl::buffer<Tscal> &buf_hpart = shambase::get_check_ref(mfield.field_hpart.get_buf());
        sycl::buffer<Tscal> &tree_field_rint = shambase::get_check_ref(
            storage.rtree_rint_field.get().get(patch_id).radix_tree_field_buf);

        sycl::range range_npart{mfield.original_elements};

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);

        u32 obj_cnt       = mfield.original_elements;
        Tscal h_tolerance = htol_up_tol;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sycl::buffer<u32> neigh_count(obj_cnt);

        shamsys::instance::get_compute_queue().wait_and_throw();

        logger::debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            sycl::accessor neigh_cnt{neigh_count, cgh, sycl::write_only, sycl::no_init};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact =
                            rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });

                neigh_cnt[id_a] = cnt;
            });
        });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            sycl::accessor scanned_neigh_cnt{pcache.scanned_cnt, cgh, sycl::read_only};
            sycl::accessor neigh{pcache.index_neigh_map, cgh, sycl::write_only, sycl::no_init};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact =
                            rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });
        });

        return pcache;
    }));

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        storage.neighbors_cache.get().preload(cur_p.id_patch);
    });

    time_neigh.end();
    storage.timings_details.neighbors += time_neigh.elasped_sec();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::reset_neighbors_cache() {
    storage.neighbors_cache.reset();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::communicate_merge_ghosts_fields() {

    StackEntry stack_loc{};

    shambase::Timer timer_interf;
    timer_interf.start();

    using namespace shamrock;
    using namespace shamrock::patch;

    bool has_alphaAV_field = solver_config.has_field_alphaAV();
    bool has_soundspeed_field = solver_config.ghost_has_soundspeed();

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz       = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint      = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint     = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart     = pdl.get_field_idx<Tscal>("hpart");

    const u32 ialpha_AV = (has_alphaAV_field) ? pdl.get_field_idx<Tscal>("alpha_AV") : 0;
    const u32 isoundspeed = (has_soundspeed_field) ? pdl.get_field_idx<Tscal>("soundspeed") : 0;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 ialpha_AV_interf =
        (has_alphaAV_field) ? ghost_layout.get_field_idx<Tscal>("alpha_AV") : 0;

    const u32 isoundspeed_interf =
        (has_soundspeed_field) ? ghost_layout.get_field_idx<Tscal>("soundspeed") : 0;

    using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
    ComputeField<Tscal> &omega                    = storage.omega.get();

    auto pdat_interf = ghost_handle.template build_interface_native<PatchData>(
        storage.ghost_patch_cache.get(),
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sycl::buffer<u32> &buf_idx, u32 cnt) {
            PatchData pdat(ghost_layout);

            pdat.reserve(cnt);

            return pdat;
        });

    ghost_handle.template modify_interface_native<PatchData>(
        storage.ghost_patch_cache.get(),
        pdat_interf,
        [&](u64 sender,
            u64,
            InterfaceBuildInfos binfo,
            sycl::buffer<u32> &buf_idx,
            u32 cnt,
            PatchData &pdat) {
            PatchData &sender_patch             = scheduler().patch_data.get_pdat(sender);
            PatchDataField<Tscal> &sender_omega = omega.get_field(sender);

            sender_patch.get_field<Tscal>(ihpart).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(ihpart_interf));
            sender_patch.get_field<Tscal>(iuint).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(iuint_interf));
            sender_patch.get_field<Tvec>(ivxyz).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tvec>(ivxyz_interf));

            sender_omega.append_subset_to(buf_idx, cnt, pdat.get_field<Tscal>(iomega_interf));

            if (has_alphaAV_field) {
                sender_patch.get_field<Tscal>(ialpha_AV).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(ialpha_AV_interf));
            }

            if (has_soundspeed_field) {
                sender_patch.get_field<Tscal>(isoundspeed).append_subset_to(
                    buf_idx, cnt, pdat.get_field<Tscal>(isoundspeed_interf));
            }
        });

    ghost_handle.template modify_interface_native<PatchData>(
        storage.ghost_patch_cache.get(),
        pdat_interf,
        [&](u64 sender,
            u64,
            InterfaceBuildInfos binfo,
            sycl::buffer<u32> &buf_idx,
            u32 cnt,
            PatchData &pdat) {
            if (sycl::length(binfo.offset_speed) > 0) {
                pdat.get_field<Tvec>(ivxyz_interf).apply_offset(binfo.offset_speed);
            }
        });

    shambase::DistributedDataShared<PatchData> interf_pdat =
        ghost_handle.communicate_pdat(ghost_layout, std::move(pdat_interf));

    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchData &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    storage.merged_patchdata_ghost.set(
        ghost_handle.template merge_native<PatchData, MergedPatchData>(
            std::move(interf_pdat),
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
                PatchData pdat_new(ghost_layout);

                u32 or_elem = pdat.get_obj_cnt();
                pdat_new.reserve(or_elem + sz_interf_map[p.id_patch]);
                u32 total_elements = or_elem;

                PatchDataField<Tscal> &cur_omega = omega.get_field(p.id_patch);

                pdat_new.get_field<Tscal>(ihpart_interf).insert(pdat.get_field<Tscal>(ihpart));
                pdat_new.get_field<Tscal>(iuint_interf).insert(pdat.get_field<Tscal>(iuint));
                pdat_new.get_field<Tvec>(ivxyz_interf).insert(pdat.get_field<Tvec>(ivxyz));
                pdat_new.get_field<Tscal>(iomega_interf).insert(cur_omega);

                if (has_alphaAV_field) {
                    pdat_new.get_field<Tscal>(ialpha_AV_interf)
                        .insert(pdat.get_field<Tscal>(ialpha_AV));
                }

                if (has_soundspeed_field) {
                    pdat_new.get_field<Tscal>(isoundspeed_interf)
                        .insert(pdat.get_field<Tscal>(isoundspeed));
                }

                pdat_new.check_field_obj_cnt_match();

                return MergedPatchData{or_elem, total_elements, std::move(pdat_new), ghost_layout};
            },
            [](MergedPatchData &mpdat, PatchData &pdat_interf) {
                mpdat.total_elements += pdat_interf.get_obj_cnt();
                mpdat.pdat.insert_elements(pdat_interf);
            }));

    timer_interf.end();
    storage.timings_details.interface += timer_interf.elasped_sec();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::reset_merge_ghosts_fields() {
    storage.merged_patchdata_ghost.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// start artificial viscosity section //////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::update_artificial_viscosity(Tscal dt) {

    sph::modules::UpdateViscosity<Tvec, Kern>(context, solver_config, storage)
        .update_artificial_viscosity(dt);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// end artificial viscosity section ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::compute_eos_fields() {

    modules::ComputeEos<Tvec, Kern>(context, solver_config, storage).compute_eos();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::reset_eos_fields() {
    storage.pressure.reset();
    storage.soundspeed.reset();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::prepare_corrector() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;
    shamrock::SchedulerUtility utility(scheduler());
    PatchDataLayout &pdl = scheduler().pdl;
    const u32 iduint     = pdl.get_field_idx<Tscal>("duint");
    const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");
    logger::debug_ln("sph::BasicGas", "save old fields");
    storage.old_axyz.set(utility.save_field<Tvec>(iaxyz, "axyz_old"));
    storage.old_duint.set(utility.save_field<Tscal>(iduint, "duint_old"));
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::update_derivs() {

    using Cfg_AV = typename Config::AVConfig;

    using None        = typename Cfg_AV::None;
    using Constant    = typename Cfg_AV::Constant;
    using VaryingMM97 = typename Cfg_AV::VaryingMM97;
    using VaryingCD10 = typename Cfg_AV::VaryingCD10;

    Cfg_AV cfg_av = solver_config.artif_viscosity;

    if (None *v = std::get_if<None>(&cfg_av.config)) {
        shambase::throw_unimplemented();
    } else if (Constant *v = std::get_if<Constant>(&cfg_av.config)) {
        update_derivs_constantAV();
    } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&cfg_av.config)) {
        update_derivs_mm97();
    } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&cfg_av.config)) {
        update_derivs_cd10();
    } else {
        shambase::throw_unimplemented();
    }

    modules::ExternalForces<Tvec, Kern> ext_forces(context, solver_config, storage);
    ext_forces.add_ext_forces();
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::update_derivs_constantAV() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shambase::DistributedData<RTree> &trees           = storage.merged_pos_trees.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    using ConfigCstAv = typename Config::AVConfig::Constant;
    ConfigCstAv *constant_av_config =
        std::get_if<ConfigCstAv>(&solver_config.artif_viscosity.config);

    if (!constant_av_config) {
        throw shambase::throw_with_loc<std::invalid_argument>(
            "cannot execute the constant viscosity kernel without constant AV config");
    }

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_axyz      = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sycl::buffer<Tscal> &buf_duint    = pdat.get_field_buf_ref<Tscal>(iduint);
        sycl::buffer<Tvec> &buf_vxyz      = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tscal> &buf_hpart    = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega    = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint     = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tscal> &buf_pressure = storage.pressure.get().get_buf_check(cur_p.id_patch);

        sycl::range range_npart{pdat.get_obj_cnt()};

        RTree &tree = trees.get(cur_p.id_patch);

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal alpha_u  = constant_av_config->alpha_u;
            const Tscal alpha_AV = constant_av_config->alpha_AV;
            const Tscal beta_AV  = constant_av_config->beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "alpha_AV :", alpha_AV);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor axyz{buf_axyz, cgh, sycl::write_only};
            sycl::accessor du{buf_duint, cgh, sycl::write_only};
            sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
            sycl::accessor omega{buf_omega, cgh, sycl::read_only};
            sycl::accessor u{buf_uint, cgh, sycl::read_only};
            sycl::accessor pressure{buf_pressure, cgh, sycl::read_only};
            sycl::accessor cs{
                storage.soundspeed.get().get_buf_check(cur_p.id_patch), cgh, sycl::read_only};

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force cte AV", [=](u64 gid) {
                u32 id_a = (u32)gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal omega_a   = omega[id_a];
                const Tscal u_a = u[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                // f32 P_a     = cs * cs * rho_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                Tscal cs_a = cs[id_a];

                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;

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

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                    Tscal P_b   = pressure[id_b];
                    // f32 P_b     = cs * cs * rho_b;
                    Tscal omega_b       = omega[id_b];
                    Tscal cs_b          = cs[id_b];

                    /////////////////
                    // internal energy update
                    //  scalar : f32  | vector : f32_3
                    const Tscal alpha_a = alpha_AV;
                    const Tscal alpha_b = alpha_AV;

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                        beta_AV,
                        alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
                });
                axyz[id_a] = force_pressure;
                du[id_a]   = tmpdU_pressure;
            });
        });
    });
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::update_derivs_mm97() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");
    const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");
    u32 ialpha_AV_interf                           = ghost_layout.get_field_idx<Tscal>("alpha_AV");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shambase::DistributedData<RTree> &trees           = storage.merged_pos_trees.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    using VarAVMM97               = typename Config::AVConfig::VaryingMM97;
    VarAVMM97 *constant_av_config = std::get_if<VarAVMM97>(&solver_config.artif_viscosity.config);

    if (!constant_av_config) {
        throw shambase::throw_with_loc<std::invalid_argument>(
            "cannot execute the constant viscosity kernel without constant AV config");
    }

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;
        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_axyz      = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sycl::buffer<Tscal> &buf_duint    = pdat.get_field_buf_ref<Tscal>(iduint);
        sycl::buffer<Tvec> &buf_vxyz      = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tscal> &buf_hpart    = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega    = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint     = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tscal> &buf_pressure = storage.pressure.get().get_buf_check(cur_p.id_patch);
        sycl::buffer<Tscal> &buf_alpha_AV = mpdat.get_field_buf_ref<Tscal>(ialpha_AV_interf);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            const Tscal pmass   = solver_config.gpart_mass;
            const Tscal alpha_u = constant_av_config->alpha_u;
            const Tscal beta_AV = constant_av_config->beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor axyz{buf_axyz, cgh, sycl::write_only};
            sycl::accessor du{buf_duint, cgh, sycl::write_only};
            sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
            sycl::accessor omega{buf_omega, cgh, sycl::read_only};
            sycl::accessor u{buf_uint, cgh, sycl::read_only};
            sycl::accessor pressure{buf_pressure, cgh, sycl::read_only};
            sycl::accessor alpha_AV{buf_alpha_AV, cgh, sycl::read_only};
            sycl::accessor cs{
                storage.soundspeed.get().get_buf_check(cur_p.id_patch), cgh, sycl::read_only};

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force MM97 AV", [=](u64 gid) {
                u32 id_a = (u32)gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal omega_a   = omega[id_a];
                const Tscal u_a = u[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                // f32 P_a     = cs * cs * rho_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                Tscal cs_a = cs[id_a];

                const Tscal alpha_a = alpha_AV[id_a];

                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;

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

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                    Tscal P_b   = pressure[id_b];
                    // f32 P_b     = cs * cs * rho_b;
                    Tscal omega_b       = omega[id_b];
                    Tscal cs_b          = cs[id_b];

                    /////////////////
                    // internal energy update
                    //  scalar : f32  | vector : f32_3
                    const Tscal alpha_b = alpha_AV[id_b];

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                        beta_AV,
                        alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
                });

                // sum_du_a               = P_a * rho_a_inv * omega_a_rho_a_inv * sum_du_a;
                // lambda_viscous_heating = -omega_a_rho_a_inv * lambda_viscous_heating;
                // lambda_shock           = lambda_viscous_heating + lambda_conductivity;
                // sum_du_a               = sum_du_a + lambda_shock;

                // out << "sum : " << sum_axyz << "\n";

                axyz[id_a] = force_pressure;
                du[id_a]   = tmpdU_pressure;
            });
        });
    });
}

template<class Tvec, template<class> class Kern>
void SPHSolve<Tvec, Kern>::update_derivs_cd10() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");
    const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");
    u32 ialpha_AV_interf                           = ghost_layout.get_field_idx<Tscal>("alpha_AV");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shambase::DistributedData<RTree> &trees           = storage.merged_pos_trees.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    using VarAVCD10               = typename Config::AVConfig::VaryingCD10;
    VarAVCD10 *constant_av_config = std::get_if<VarAVCD10>(&solver_config.artif_viscosity.config);

    if (!constant_av_config) {
        throw shambase::throw_with_loc<std::invalid_argument>(
            "cannot execute the constant viscosity kernel without constant AV config");
    }

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;
        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_axyz      = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sycl::buffer<Tscal> &buf_duint    = pdat.get_field_buf_ref<Tscal>(iduint);
        sycl::buffer<Tvec> &buf_vxyz      = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tscal> &buf_hpart    = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega    = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint     = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tscal> &buf_pressure = storage.pressure.get().get_buf_check(cur_p.id_patch);
        sycl::buffer<Tscal> &buf_alpha_AV = mpdat.get_field_buf_ref<Tscal>(ialpha_AV_interf);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            const Tscal pmass   = solver_config.gpart_mass;
            const Tscal alpha_u = constant_av_config->alpha_u;
            const Tscal beta_AV = constant_av_config->beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor axyz{buf_axyz, cgh, sycl::write_only};
            sycl::accessor du{buf_duint, cgh, sycl::write_only};
            sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
            sycl::accessor omega{buf_omega, cgh, sycl::read_only};
            sycl::accessor u{buf_uint, cgh, sycl::read_only};
            sycl::accessor pressure{buf_pressure, cgh, sycl::read_only};
            sycl::accessor alpha_AV{buf_alpha_AV, cgh, sycl::read_only};
            sycl::accessor cs{
                storage.soundspeed.get().get_buf_check(cur_p.id_patch), cgh, sycl::read_only};

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force CD10 AV", [=](u64 gid) {
                u32 id_a = (u32)gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a           = hpart[id_a];
                Tvec xyz_a          = xyz[id_a];
                Tvec vxyz_a         = vxyz[id_a];
                Tscal P_a           = pressure[id_a];
                Tscal cs_a = cs[id_a];
                Tscal omega_a       = omega[id_a];
                const Tscal u_a     = u[id_a];
                const Tscal alpha_a = alpha_AV[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                // f32 P_a     = cs * cs * rho_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);


                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    // compute only omega_a
                    Tvec dr    = xyz_a - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    Tvec vxyz_b         = vxyz[id_b];
                    const Tscal u_b     = u[id_b];
                    Tscal P_b           = pressure[id_b];
                    Tscal omega_b       = omega[id_b];
                    const Tscal alpha_b = alpha_AV[id_b];
                    Tscal cs_b  = cs[id_b];

                    Tscal rab = sycl::sqrt(rab2);

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                        beta_AV,
                        alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
                });

                axyz[id_a] = force_pressure;
                du[id_a]   = tmpdU_pressure;
            });
        });
    });
}

template<class Tvec, template<class> class Kern>
bool SPHSolve<Tvec, Kern>::apply_corrector(Tscal dt, u64 Npart_all) {
    return false;
}

template<class Tvec, template<class> class Kern>
auto SPHSolve<Tvec, Kern>::evolve_once(
    Tscal t_current, Tscal dt, bool do_dump, std::string vtk_dump_name, bool vtk_dump_patch_id)
    -> Tscal {
    StackEntry stack_loc{};

    struct DumpOption {
        bool vtk_do_dump;
        std::string vtk_dump_fname;
        bool vtk_dump_patch_id;
    };

    DumpOption dump_opt{do_dump, vtk_dump_name, vtk_dump_patch_id};

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("sph::Model", shambase::format("t = {}, dt = {}", t_current, dt));
    }

    shambase::Timer tstep;
    tstep.start();

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");

    shamrock::SchedulerUtility utility(scheduler());

    modules::SinkParticlesUpdate<Tvec, Kern> sink_update(context, solver_config, storage);
    modules::ExternalForces<Tvec, Kern> ext_forces(context, solver_config, storage);
    
    sink_update.accrete_particles();
    ext_forces.point_mass_accrete_particles();


    do_predictor_leapfrog(dt);

    update_artificial_viscosity(dt);

    sink_update.predictor_step(dt);


    sink_update.compute_ext_forces();

    
    ext_forces.compute_ext_forces_indep_v();

    gen_serial_patch_tree();

    apply_position_boundary(t_current);

    u64 Npart_all = scheduler().get_total_obj_count();

    sph_prestep(t_current);

    using RTree = RadixTree<u_morton, Tvec>;

    SPHSolverImpl solver(context);

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
    auto &merged_xyzh                             = storage.merged_xyzh.get();
    shambase::DistributedData<RTree> &trees       = storage.merged_pos_trees.get();
    ComputeField<Tscal> &omega                    = storage.omega.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    using RTreeField = RadixTreeField<Tscal>;
    shambase::DistributedData<RTreeField> rtree_field_h;

    Tscal next_cfl = 0;

    u32 corrector_iter_cnt    = 0;
    bool need_rerun_corrector = false;
    do {

        reset_merge_ghosts_fields();
        reset_eos_fields();

        if (corrector_iter_cnt == 50) {
            throw shambase::throw_with_loc<std::runtime_error>(
                "the corrector has made over 50 loops, either their is a bug, either you are using "
                "a dt that is too large");
        }

        // communicate fields
        communicate_merge_ghosts_fields();

        // compute pressure
        compute_eos_fields();

        // compute force
        logger::debug_ln("sph::BasicGas", "compute force");

        // save old acceleration
        prepare_corrector();

        update_derivs();

        modules::ConservativeCheck<Tvec, Kern> cv_check(context, solver_config, storage);
        cv_check.check_conservation();

        ComputeField<Tscal> vepsilon_v_sq =
            utility.make_compute_field<Tscal>("vmean epsilon_v^2", 1);
        ComputeField<Tscal> uepsilon_u_sq =
            utility.make_compute_field<Tscal>("umean epsilon_u^2", 1);

        // corrector
        logger::debug_ln("sph::BasicGas", "leapfrog corrector");
        utility.fields_leapfrog_corrector<Tvec>(
            ivxyz, iaxyz, storage.old_axyz.get(), vepsilon_v_sq, dt / 2);
        utility.fields_leapfrog_corrector<Tscal>(
            iuint, iduint, storage.old_duint.get(), uepsilon_u_sq, dt / 2);

        storage.old_axyz.reset();
        storage.old_duint.reset();

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

        logger::debug_ln("BasicGas", "epsilon v :", eps_v);

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

            // logger::info_ln("rerun corrector ...");
        } else {
            need_rerun_corrector = false;
        }

        if (!need_rerun_corrector) {

            sink_update.corrector_step(dt);

            logger::debug_ln("BasicGas", "computing next CFL");

            ComputeField<Tscal> vsig_max_dt = utility.make_compute_field<Tscal>("vsig_a", 1);

            shambase::DistributedData<MergedPatchData> &mpdat =
                storage.merged_patchdata_ghost.get();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
                PatchData &mpdat              = merged_patch.pdat;

                sycl::buffer<Tvec> &buf_xyz =
                    shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
                sycl::buffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
                sycl::buffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
                sycl::buffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
                sycl::buffer<Tscal> &buf_pressure =
                    storage.pressure.get().get_buf_check(cur_p.id_patch);
                sycl::buffer<Tscal> &vsig_buf = vsig_max_dt.get_buf_check(cur_p.id_patch);

                sycl::range range_npart{pdat.get_obj_cnt()};

                tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

                /////////////////////////////////////////////

                {
                    NamedStackEntry tmppp{"compute vsig"};
                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        const Tscal pmass    = solver_config.gpart_mass;
                        const Tscal alpha_u  = 1.0;
                        const Tscal alpha_AV = 1.0;
                        const Tscal beta_AV  = 2.0;

                        tree::ObjectCacheIterator particle_looper(pcache, cgh);

                        sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                        sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
                        sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                        sycl::accessor u{buf_uint, cgh, sycl::read_only};
                        sycl::accessor pressure{buf_pressure, cgh, sycl::read_only};


                        sycl::accessor cs{
                            storage.soundspeed.get().get_buf_check(cur_p.id_patch), cgh, sycl::read_only};

                        sycl::accessor vsig{vsig_buf, cgh, sycl::write_only, sycl::no_init};

                        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                        shambase::parralel_for(
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
                }
            });

            ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", 1);

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);

                sycl::buffer<Tvec> &buf_axyz =
                    shambase::get_check_ref(pdat.get_field<Tvec>(iaxyz).get_buf());
                sycl::buffer<Tscal> &buf_hpart = shambase::get_check_ref(
                    merged_patch.pdat.get_field<Tscal>(ihpart_interf).get_buf());
                sycl::buffer<Tscal> &vsig_buf   = vsig_max_dt.get_buf_check(cur_p.id_patch);
                sycl::buffer<Tscal> &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                    sycl::accessor a{buf_axyz, cgh, sycl::read_only};
                    sycl::accessor vsig{vsig_buf, cgh, sycl::read_only};
                    sycl::accessor cfl_dt{cfl_dt_buf, cgh, sycl::write_only, sycl::no_init};

                    Tscal C_cour  = solver_config.cfl_cour;
                    Tscal C_force = solver_config.cfl_force;

                    cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                        Tscal h_a     = hpart[item];
                        Tscal vsig_a  = vsig[item];
                        Tscal abs_a_a = sycl::length(a[item]);

                        Tscal dt_c = C_cour * h_a / vsig_a;
                        Tscal dt_f = C_force * sycl::sqrt(h_a / abs_a_a);

                        cfl_dt[item] = sycl::min(dt_c, dt_f);
                    });
                });
            });

            Tscal rank_dt = cfl_dt.compute_rank_min();

            logger::debug_ln(
                "BasigGas", "rank", shamcomm::world_rank(), "found cfl dt =", rank_dt);

            next_cfl = shamalgs::collective::allreduce_min(rank_dt);

            if (shamcomm::world_rank() == 0) {
                logger::info_ln("sph::Model", "cfl dt =", next_cfl);
            }

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
                    .update_dtdivv();
            }

            // this should not be needed idealy, but we need the pressure on the ghosts and 
            // we don't want to communicate it as it can be recomputed from the other fields
            // hence we copy the soundspeed at the end of the step to a field in the patchdata
            if (solver_config.has_field_soundspeed()) {
                const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");
                scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
                    sycl::buffer<Tscal> &buf_cs    = pdat.get_field_buf_ref<Tscal>(isoundspeed);

                    sycl::range range_npart{pdat.get_obj_cnt()};

                    /////////////////////////////////////////////

                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        const Tscal pmass = solver_config.gpart_mass;

                        sycl::accessor cs_in{
                            storage.soundspeed.get().get_buf_check(cur_p.id_patch), cgh, sycl::read_only};
                        sycl::accessor cs{buf_cs, cgh, sycl::write_only, sycl::no_init};

                        cgh.parallel_for(
                            sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                                cs[item]   = cs_in[item];
                            });
                    });
                });
            }

        } // if (!need_rerun_corrector) {

        corrector_iter_cnt++;

    } while (need_rerun_corrector);

    reset_merge_ghosts_fields();
    reset_eos_fields();

    // if delta too big jump to compute force

    if (dump_opt.vtk_do_dump) {

        shambase::Timer timer_io;
        timer_io.start();

        ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho", 1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_h{
                    shambase::get_check_ref(pdat.get_field<Tscal>(ihpart).get_buf()),
                    cgh,
                    sycl::read_only};

                sycl::accessor acc_rho{
                    shambase::get_check_ref(density.get_buf(p.id_patch)),
                    cgh,
                    sycl::write_only,
                    sycl::no_init};
                const Tscal part_mass = solver_config.gpart_mass;

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid = (u32)item.get_id();
                    using namespace shamrock::sph;
                    Tscal rho_ha = rho_h(part_mass, acc_h[gid], Kernel::hfactd);
                    acc_rho[gid] = rho_ha;
                });
            });
        });

        shamrock::LegacyVtkWritter writter = start_dump<Tvec>(scheduler(), dump_opt.vtk_dump_fname);
        writter.add_point_data_section();

        u32 fnum = 0;
        if (dump_opt.vtk_dump_patch_id) {
            fnum += 2;
        }
        fnum++;
        fnum++;
        fnum++;
        fnum++;
        fnum++;
        fnum++;

        if (solver_config.has_field_alphaAV()) {
            fnum++;
        }

        if (solver_config.has_field_divv()) {
            fnum++;
        }

        if (solver_config.has_field_curlv()) {
            fnum++;
        }

        if (solver_config.has_field_soundspeed()) {
            fnum++;
        }

        if (solver_config.has_field_dtdivv()) {
            fnum++;
        }

        writter.add_field_data_section(fnum);

        if (dump_opt.vtk_dump_patch_id) {
            vtk_dump_add_patch_id(scheduler(), writter);
            vtk_dump_add_worldrank(scheduler(), writter);
        }

        vtk_dump_add_field<Tscal>(scheduler(), writter, ihpart, "h");
        vtk_dump_add_field<Tscal>(scheduler(), writter, iuint, "u");
        vtk_dump_add_field<Tvec>(scheduler(), writter, ivxyz, "v");
        vtk_dump_add_field<Tvec>(scheduler(), writter, iaxyz, "a");

        if (solver_config.has_field_alphaAV()) {
            const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");
            vtk_dump_add_field<Tscal>(scheduler(), writter, ialpha_AV, "alpha_AV");
        }

        if (solver_config.has_field_divv()) {
            const u32 idivv = pdl.get_field_idx<Tscal>("divv");
            vtk_dump_add_field<Tscal>(scheduler(), writter, idivv, "divv");
        }

        if (solver_config.has_field_dtdivv()) {
            const u32 idtdivv = pdl.get_field_idx<Tscal>("dtdivv");
            vtk_dump_add_field<Tscal>(scheduler(), writter, idtdivv, "dtdivv");
        }

        if (solver_config.has_field_curlv()) {
            const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");
            vtk_dump_add_field<Tvec>(scheduler(), writter, icurlv, "curlv");
        }

        if (solver_config.has_field_soundspeed()) {
            const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");
            vtk_dump_add_field<Tscal>(scheduler(), writter, isoundspeed, "soundspeed");
        }

        vtk_dump_add_compute_field(scheduler(), writter, density, "rho");
        vtk_dump_add_compute_field(scheduler(), writter, omega, "omega");

        timer_io.end();
        storage.timings_details.io += timer_io.elasped_sec();
    }

    tstep.end();

    u64 rank_count = scheduler().get_rank_count();
    f64 rate       = f64(rank_count) / tstep.elasped_sec();

    // logger::info_ln("SPHSolver", "process rate : ", rate, "particle.s-1");

    std::string log_rank_rate = shambase::format(
        "\n| {:<4} |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:3.0f} % |",
        shamcomm::world_rank(),
        rate,
        rank_count,
        tstep.elasped_sec(),
        100 * (storage.timings_details.interface / tstep.elasped_sec()),
        100 * (storage.timings_details.neighbors / tstep.elasped_sec()),
        100 * (storage.timings_details.io / tstep.elasped_sec()));

    std::string gathered = "";
    shamcomm::gather_str(log_rank_rate, gathered);

    if (shamcomm::world_rank() == 0) {
        std::string print = "processing rate infos : \n";
        print +=
            ("---------------------------------------------------------------------------------\n");
        print += ("| rank |  rate  (N.s^-1)  |      N      | t compute (s) | interf | neigh |   io "
                  " |\n");
        print +=
            ("---------------------------------------------------------------------------------");
        print += (gathered) + "\n";
        print +=
            ("---------------------------------------------------------------------------------");
        logger::info_ln("sph::Model", print);
    }

    storage.timings_details.reset();

    reset_serial_patch_tree();
    reset_ghost_handler();
    storage.merged_xyzh.reset();
    storage.omega.reset();
    clear_merged_pos_trees();
    clear_ghost_cache();
    reset_presteps_rint();
    reset_neighbors_cache();

    return next_cfl;
}

using namespace shammath;

template class shammodels::sph::Solver<f64_3, M4>;
template class shammodels::sph::Solver<f64_3, M6>;