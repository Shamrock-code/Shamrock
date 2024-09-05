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
 *
 */

#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shamcomm/collectives.hpp"
#include "shammodels/amr/basegodunov/modules/AMRGraphGen.hpp"
#include "shammodels/amr/basegodunov/modules/AMRTree.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeCFL.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeCellInfos.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeFlux.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeGradient.hpp"
#include "shammodels/amr/basegodunov/modules/ComputeTimeDerivative.hpp"
#include "shammodels/amr/basegodunov/modules/ConsToPrim.hpp"
#include "shammodels/amr/basegodunov/modules/FaceInterpolate.hpp"
#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"
#include "shammodels/amr/basegodunov/modules/StencilGenerator.hpp"
#include "shammodels/amr/basegodunov/modules/TimeIntegrator.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::evolve_once() {
    Tscal t_current = solver_config.get_time();
    Tscal dt_input  = solver_config.get_dt();

    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("amr::Godunov", shambase::format("t = {}, dt = {}", t_current, dt_input));
    }

    shambase::Timer tstep;
    tstep.start();

    scheduler().update_local_load_value([&](shamrock::patch::Patch p) {
        return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    // ghost zone exchange
    modules::GhostZones gz(context, solver_config, storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();

    modules::ComputeCellInfos comp_cell_infos(context, solver_config, storage);
    comp_cell_infos.compute_aabb();

    // compute bound received
    // round to next pow of 2
    // build radix trees
    modules::AMRTree amrtree(context, solver_config, storage);
    amrtree.build_trees();

    amrtree.correct_bounding_box();

    // modules::StencilGenerator stencil_gen(context,solver_config,storage);
    // stencil_gen.make_stencil();

    modules::AMRGraphGen graph_gen(context, solver_config, storage);
    auto block_oriented_graph = graph_gen.find_AMR_block_graph_links_common_face();

    graph_gen.lower_AMR_block_graph_to_cell_common_face_graph(block_oriented_graph);

    // compute prim variable
    modules::ConsToPrim ctop(context, solver_config, storage);
    ctop.cons_to_prim();

    // compute & limit gradients
    modules::ComputeGradient grad_compute(context, solver_config, storage);
    grad_compute.compute_grad_rho_van_leer();
    grad_compute.compute_grad_v_van_leer();
    grad_compute.compute_grad_P_van_leer();
    if (solver_config.is_dust_on()) {
        grad_compute.compute_grad_rho_dust_van_leer();
        grad_compute.compute_grad_v_dust_van_leer();
    }

    // shift values
    modules::FaceInterpolate face_interpolator(context, solver_config, storage);
    Tscal dt_face_interp = 0;
    if (solver_config.face_half_time_interpolation) {
        dt_face_interp = dt_input / 2.0;
    }
    face_interpolator.interpolate_rho_to_face(dt_face_interp);
    face_interpolator.interpolate_v_to_face(dt_face_interp);
    face_interpolator.interpolate_P_to_face(dt_face_interp);

    if (solver_config.is_dust_on()) {
        face_interpolator.interpolate_rho_dust_to_face(dt_face_interp);
        face_interpolator.interpolate_v_dust_to_face(dt_face_interp);
    }

    // flux
    modules::ComputeFlux flux_compute(context, solver_config, storage);
    flux_compute.compute_flux();
    if (solver_config.is_dust_on()) {
        flux_compute.compute_flux_dust();
    }
    // compute dt fields
    modules::ComputeTimeDerivative dt_compute(context, solver_config, storage);
    dt_compute.compute_dt_fields();
    // TODO call compute_dt_dust when dust_config is on

    // RK2 + flux lim

    modules::TimeIntegrator dt_integ(context, solver_config, storage);
    dt_integ.forward_euler(dt_input);

    if (false) {
        static u32 cnt_debug = 0;
        do_debug_vtk_dump(shambase::format("debug_dump_{:04}.vtk", cnt_debug));
        cnt_debug++;
    }

    modules::ComputeCFL cfl_compute(context, solver_config, storage);
    f64 new_dt = cfl_compute.compute_cfl();

    solver_config.set_next_dt(new_dt);
    solver_config.set_time(t_current + dt_input);

    storage.dtrho.reset();
    storage.dtrhov.reset();
    storage.dtrhoe.reset();

    storage.flux_rho_face_xp.reset();
    storage.flux_rho_face_xm.reset();
    storage.flux_rho_face_yp.reset();
    storage.flux_rho_face_ym.reset();
    storage.flux_rho_face_zp.reset();
    storage.flux_rho_face_zm.reset();
    storage.flux_rhov_face_xp.reset();
    storage.flux_rhov_face_xm.reset();
    storage.flux_rhov_face_yp.reset();
    storage.flux_rhov_face_ym.reset();
    storage.flux_rhov_face_zp.reset();
    storage.flux_rhov_face_zm.reset();
    storage.flux_rhoe_face_xp.reset();
    storage.flux_rhoe_face_xm.reset();
    storage.flux_rhoe_face_yp.reset();
    storage.flux_rhoe_face_ym.reset();
    storage.flux_rhoe_face_zp.reset();
    storage.flux_rhoe_face_zm.reset();

    storage.rho_face_xp.reset();
    storage.rho_face_xm.reset();
    storage.rho_face_yp.reset();
    storage.rho_face_ym.reset();
    storage.rho_face_zp.reset();
    storage.rho_face_zm.reset();

    storage.vel_face_xp.reset();
    storage.vel_face_xm.reset();
    storage.vel_face_yp.reset();
    storage.vel_face_ym.reset();
    storage.vel_face_zp.reset();
    storage.vel_face_zm.reset();

    storage.press_face_xp.reset();
    storage.press_face_xm.reset();
    storage.press_face_yp.reset();
    storage.press_face_ym.reset();
    storage.press_face_zp.reset();
    storage.press_face_zm.reset();

    storage.grad_rho.reset();
    storage.dx_v.reset();
    storage.dy_v.reset();
    storage.dz_v.reset();
    storage.grad_P.reset();

    storage.vel.reset();
    storage.press.reset();

    storage.cell_infos.reset();
    storage.cell_link_graph.reset();

    storage.trees.reset();
    storage.merge_patch_bounds.reset();

    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();
    storage.ghost_zone_infos.reset();

    storage.serial_patch_tree.reset();

    tstep.end();

    u64 rank_count = scheduler().get_rank_count() * AMRBlock::block_size;
    f64 rate       = f64(rank_count) / tstep.elasped_sec();

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
        print += ("--------------------------------------------------------------------------------"
                  "-\n");
        print += ("| rank |  rate  (N.s^-1)  |      N      | t compute (s) | interf | neigh |   io "
                  " |\n");
        print += ("--------------------------------------------------------------------------------"
                  "-");
        print += (gathered) + "\n";
        print += ("--------------------------------------------------------------------------------"
                  "-");
        logger::info_ln("amr::Godunov", print);
    }

    storage.timings_details.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::do_debug_vtk_dump(std::string filename) {

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    u32 block_size = Solver::AMRBlock::block_size;

    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<TgridVec>> pos1 = sched.rankgather_field<TgridVec>(0);
    std::unique_ptr<sycl::buffer<TgridVec>> pos2 = sched.rankgather_field<TgridVec>(1);

    sycl::buffer<Tvec> pos_min_cell(num_obj * block_size);
    sycl::buffer<Tvec> pos_max_cell(num_obj * block_size);

    shamsys::instance::get_compute_queue().submit([&, block_size](sycl::handler &cgh) {
        sycl::accessor acc_p1{shambase::get_check_ref(pos1), cgh, sycl::read_only};
        sycl::accessor acc_p2{shambase::get_check_ref(pos2), cgh, sycl::read_only};
        sycl::accessor cell_min{pos_min_cell, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor cell_max{pos_max_cell, cgh, sycl::write_only, sycl::no_init};

        using Block = typename Solver::AMRBlock;

        shambase::parralel_for(cgh, num_obj, "rescale cells", [=](u64 id_a) {
            Tvec block_min = acc_p1[id_a].template convert<Tscal>();
            Tvec block_max = acc_p2[id_a].template convert<Tscal>();

            Tvec delta_cell = (block_max - block_min) / Block::side_size;
#pragma unroll
            for (u32 ix = 0; ix < Block::side_size; ix++) {
#pragma unroll
                for (u32 iy = 0; iy < Block::side_size; iy++) {
#pragma unroll
                    for (u32 iz = 0; iz < Block::side_size; iz++) {
                        u32 i                           = Block::get_index({ix, iy, iz});
                        Tvec delta_val                  = delta_cell * Tvec{ix, iy, iz};
                        cell_min[id_a * block_size + i] = block_min + delta_val;
                        cell_max[id_a * block_size + i] = block_min + (delta_cell) + delta_val;
                    }
                }
            }
        });
    });

    writer.write_voxel_cells(pos_min_cell, pos_max_cell, num_obj * block_size);

    writer.add_cell_data_section();
    writer.add_field_data_section(11);

    std::unique_ptr<sycl::buffer<Tscal>> fields_rho = sched.rankgather_field<Tscal>(2);
    writer.write_field("rho", fields_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> fields_vel = sched.rankgather_field<Tvec>(3);
    writer.write_field("rhovel", fields_vel, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> fields_eint = sched.rankgather_field<Tscal>(4);
    writer.write_field("rhoetot", fields_eint, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> grad_rho
        = storage.grad_rho.get().rankgather_computefield(sched);
    writer.write_field("grad_rho", grad_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dx_v = storage.dx_v.get().rankgather_computefield(sched);
    writer.write_field("dx_v", dx_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dy_v = storage.dy_v.get().rankgather_computefield(sched);
    writer.write_field("dy_v", dy_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dz_v = storage.dz_v.get().rankgather_computefield(sched);
    writer.write_field("dz_v", dz_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> grad_P
        = storage.grad_P.get().rankgather_computefield(sched);
    writer.write_field("grad_P", grad_P, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrho = storage.dtrho.get().rankgather_computefield(sched);
    writer.write_field("dtrho", dtrho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dtrhov
        = storage.dtrhov.get().rankgather_computefield(sched);
    writer.write_field("dtrhov", dtrhov, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrhoe
        = storage.dtrhoe.get().rankgather_computefield(sched);
    writer.write_field("dtrhoe", dtrhoe, num_obj * block_size);
}

template class shammodels::basegodunov::Solver<f64_3, i64_3>;
