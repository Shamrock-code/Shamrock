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
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
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
#include "shammodels/sph/modules/ComputeLoadBalanceValue.hpp"
#include "shammodels/sph/modules/ConservativeCheck.hpp"
#include "shammodels/sph/modules/DiffOperator.hpp"
#include "shammodels/sph/modules/DiffOperatorDtDivv.hpp"
#include "shammodels/sph/modules/ExternalForces.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shammodels/sph/modules/UpdateDerivs.hpp"
#include "shammodels/sph/modules/UpdateViscosity.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
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

    auto &merged_xyzh         = storage.merged_xyzh.get();


    shambase::DistributedData<RTree> trees =
        merged_xyzh.template map<RTree>([&](u64 id, PreStepMergedField &merged) {
            Tvec bmin = merged.bounds.lower;
            Tvec bmax = merged.bounds.upper;

            RTree tree(shamsys::instance::get_compute_queue(),
                        {bmin, bmax},
                        merged.field_pos.get_buf(),
                        merged.field_pos.get_obj_cnt(),
                        solver_config.tree_reduction_level);

            return tree;
        });

    trees.for_each([&](u64 id, RTree &tree) {
        tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
        tree.convert_bounding_box(shamsys::instance::get_compute_queue());
    });



    bool corect_boxes = solver_config.use_two_stage_search;
    if(corect_boxes){

        trees.for_each([&](u64 id, RTree & tree){
            
            u32 leaf_count = tree.tree_reduced_morton_codes.tree_leaf_count;
            u32 internal_cell_count = tree.tree_struct.internal_cell_count;
            u32 tot_count = leaf_count + internal_cell_count;

            sycl::buffer<Tvec> tmp_min_cell(tot_count);
            sycl::buffer<Tvec> tmp_max_cell(tot_count);

            sycl::buffer<Tvec> &buf_part_pos = shambase::get_check_ref(merged_xyzh.get(id).field_pos.get_buf());

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator cell_looper(tree, cgh);

                u32 leaf_offset = tree.tree_struct.internal_cell_count;

                sycl::accessor acc_pos {buf_part_pos, cgh, sycl::read_only};

                sycl::accessor comp_min {tmp_min_cell, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor comp_max {tmp_max_cell, cgh, sycl::write_only, sycl::no_init};


                // how to lose a f****ing afternoon :
                // 1 - code a nice algorithm that should optimize the code
                // 2 - pass all the tests
                // 3 - benchmark it and discover big loss in perf for no reasons
                // 4 - change a parameter and discover a segfault (on GPU to have more fun ....) 
                // 5 - find that actually the core algorithm of the code create a bug in the new thing
                // 6 - discover that every value in everything is wrong
                // 7 - spent the whole night on it
                // 8 - start putting prints everywhere
                // 9 - isolate a bugged id
                // 10 - try to understand why a f***ing leaf is as big as the root of the tree
                // 11 - **** a few hours latter
                // 12 - the goddam c++ standard define std::numeric_limits<float>::min() to be epsilon instead of -max
                // 13 - road rage 
                // 14 - open a bier
                // alt f4 the ide

                Tvec imin = shambase::VectorProperties<Tvec>::get_max();
                Tvec imax = -shambase::VectorProperties<Tvec>::get_max();

                shambase::parralel_for(cgh, leaf_count,"compute leaf boxes", [=](u64 leaf_id){

                    Tvec min = imin;
                    Tvec max = imax;

                    cell_looper.iter_object_in_cell(leaf_id + leaf_offset, [&](u32 part_id){
                        Tvec r = acc_pos[part_id];

                        min = sham::min(min, r);
                        max = sham::max(max, r);
                    });

                    comp_min[leaf_offset + leaf_id] = min;
                    comp_max[leaf_offset + leaf_id] = max;

                });

            });

            //{
            //    u32 leaf_offset = tree.tree_struct.internal_cell_count;
            //    sycl::host_accessor pos_min_cell  {tmp_min_cell};
            //    sycl::host_accessor pos_max_cell  {tmp_max_cell};
            //    
            //    for (u32 i = 0; i < 1000; i++) {
            //            logger::raw_ln(i,pos_max_cell[i+leaf_offset] - pos_min_cell[i+leaf_offset]);
            //        
            //    }
            //}

            auto ker_reduc_hmax = [&](sycl::handler &cgh) {
                u32 offset_leaf = internal_cell_count;

                sycl::accessor comp_min {tmp_min_cell, cgh, sycl::read_write};
                sycl::accessor comp_max {tmp_max_cell, cgh, sycl::read_write};

                sycl::accessor rchild_id   {shambase::get_check_ref(tree.tree_struct.buf_rchild_id  ),cgh, sycl::read_only};
                sycl::accessor lchild_id   {shambase::get_check_ref(tree.tree_struct.buf_lchild_id  ),cgh, sycl::read_only};
                sycl::accessor rchild_flag {shambase::get_check_ref(tree.tree_struct.buf_rchild_flag),cgh, sycl::read_only};
                sycl::accessor lchild_flag {shambase::get_check_ref(tree.tree_struct.buf_lchild_flag),cgh, sycl::read_only};

                shambase::parralel_for(cgh, internal_cell_count,"propagate up", [=](u64 gid){

                    u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                    u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                    Tvec bminl = comp_min[lid];
                    Tvec bminr = comp_min[rid];
                    Tvec bmaxl = comp_max[lid];
                    Tvec bmaxr = comp_max[rid];

                    Tvec bmin = shambase::sycl_utils::g_sycl_min(bminl, bminr);
                    Tvec bmax = shambase::sycl_utils::g_sycl_max(bmaxl, bmaxr);

                    comp_min[gid] = bmin;
                    comp_max[gid] = bmax;
                });
            };

            for (u32 i = 0; i < tree.tree_depth; i++) {
                shamsys::instance::get_compute_queue().submit(ker_reduc_hmax);
            }

            sycl::buffer<Tvec> & tree_bmin = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
            sycl::buffer<Tvec> & tree_bmax = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator cell_looper(tree, cgh);

                u32 leaf_offset = tree.tree_struct.internal_cell_count;

                sycl::accessor comp_bmin {tmp_min_cell, cgh, sycl::read_only};
                sycl::accessor comp_bmax {tmp_max_cell, cgh, sycl::read_only};

                sycl::accessor tree_buf_min {tree_bmin, cgh, sycl::read_write};
                sycl::accessor tree_buf_max {tree_bmax, cgh, sycl::read_write};

                shambase::parralel_for(cgh, tot_count,"write in tree range", [=](u64 nid){

                    Tvec load_min = comp_bmin[nid]; 
                    Tvec load_max = comp_bmax[nid]; 

                    tree_buf_min[nid] = load_min;
                    tree_buf_max[nid] = load_max;   

                });

            });

        });
    }


    
    storage.merged_pos_trees.set(std::move(trees));
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
    u32 hstep_max = 100;
    for (; hstep_cnt < hstep_max; hstep_cnt++) {

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
            
            logger::debug_ln("Smoothinglength","iteration :",iter_h, "epsmax",max_eps_h);

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

    if (hstep_cnt == hstep_max) {
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
    if(solver_config.use_two_stage_search){
        shammodels::sph::modules::NeighbourCache<Tvec,u_morton, Kern>(context, solver_config, storage)
        .start_neighbors_cache_2stages();
    }else{
        shammodels::sph::modules::NeighbourCache<Tvec,u_morton, Kern>(context, solver_config, storage)
        .start_neighbors_cache();
    }
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

    modules::UpdateDerivs<Tvec, Kern> derivs(context, solver_config, storage);
    derivs.update_derivs();

    modules::ExternalForces<Tvec, Kern> ext_forces(context, solver_config, storage);
    ext_forces.add_ext_forces();
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

    modules::ComputeLoadBalanceValue<Tvec, Kern> (context, solver_config, storage).update_load_balancing();

    scheduler().scheduler_step(true, true);

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



            //if (solver_config.has_field_divv()) {
            //    sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
            //        .update_divv();
            //}
            //
            //if (solver_config.has_field_curlv()) {
            //    sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
            //        .update_curlv();
            //}
            //
            //if (solver_config.has_field_dtdivv()) {
            //    sph::modules::DiffOperatorDtDivv<Tvec, Kern>(context, solver_config, storage)
            //        .update_dtdivv(false);
            //}

            if (solver_config.has_field_dtdivv()) {
                
                if(solver_config.combined_dtdiv_divcurlv_compute){
                    //L2 distance r :  1.32348e-05 
                    //L2 distance h :  7.68714e-06 
                    //L2 distance vr :  0.0133985 
                    //L2 distance u :  0.0380309 
                    if (solver_config.has_field_dtdivv()) {     
                        sph::modules::DiffOperatorDtDivv<Tvec, Kern>(context, solver_config, storage)
                            .update_dtdivv(true);
                    }
                }else{
                    //L2 distance r :  1.32348e-05 
                    //L2 distance h :  7.68714e-06 
                    //L2 distance vr :  0.0133985 
                    //L2 distance u :  0.0380309 

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

            }else{
                if (solver_config.has_field_divv()) {
                    sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
                        .update_divv();
                }

                if (solver_config.has_field_curlv()) {
                    sph::modules::DiffOperators<Tvec, Kern>(context, solver_config, storage)
                        .update_curlv();
                }
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
