// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


/**
 * @file DragIntegrator.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/amr/basegodunov/modules/DragIntegrator.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::involve_with_no_src(Tscal dt){

   StackEntry stack_lock{};

   using namespace shamrock::patch;
   using namespace shamrock;
   using namespace shammath;

   const u32 ndust  = solver_config.dust_config.ndust;

   SchedulerUtility utility(scheduler());
   shamrock::ComputeField<Tscal> cfield_rho_next_bf_drag   = utility.make_compute_field<Tscal>("rho_next_bf_drag", AMRBlock::block_size);
   shamrock::ComputeField<Tvec>  cfield_rhov_next_bf_drag  = utility.make_compute_field<Tvec>("rhov_next_bf_drag", AMRBlock::block_size);
   shamrock::ComputeField<Tscal> cfield_rhoe_next_bf_drag  = utility.make_compute_field<Tscal>("rhoe_next_bf_drag" , AMRBlock::block_size);
   shamrock::ComputeField<Tscal> cfield_rho_d_next_bf_drag   = utility.make_compute_field<Tscal>("rho_d_next_bf_drag", ndust * AMRBlock::block_size);
   shamrock::ComputeField<Tvec>  cfield_rhov_d_next_bf_drag  = utility.make_compute_field<Tvec>("rhov_d_next_bf_drag", ndust * AMRBlock::block_size);


   shamrock::ComputeField<Tscal> &cfield_dtrho   = storage.dtrho.get();
   shamrock::ComputeField<Tvec>  &cfield_dtrhov  = storage.dtrhov.get();
   shamrock::ComputeField<Tscal> &cfield_dtrhoe  = storage.dtrhoe.get();
   shamrock::ComputeField<Tscal> &cfield_dtrho_d = storage.dtrho_dust.get();
   shamrock::ComputeField<Tvec> &cfield_dtrhov_d = storage.dtrhov_dust.get();

    scheduler().for_each_patchdata_nonempty([&,dt, ndust] (const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
        logger::debug_ln("[AMR evolve time step before drag ]", "evolve field with no drag patch", p.id_patch);

        sycl::queue &q          = shamsys::instance::get_compute_queue();
        u32 id                  = p.id_patch;

        sycl::buffer<Tscal> &dt_rho_patch    = cfield_dtrho.get_buf_check(id);
        sycl::buffer<Tvec>  &dt_rhov_patch   = cfield_dtrhov.get_buf_check(id);
        sycl::buffer<Tscal> &dt_rhoe_patch   = cfield_dtrhoe.get_buf_check(id);
        sycl::buffer<Tscal> &dt_rho_d_patch  = cfield_dtrho_d.get_buf_check(id);
        sycl::buffer<Tvec>  &dt_rhov_d_patch = cfield_dtrhov_d.get_buf_check(id);

        sycl::buffer<Tscal> &rho_patch    = cfield_rho_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tvec>  &rhov_patch   = cfield_rhov_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tscal> &rhoe_patch   = cfield_rhoe_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tscal> &rho_d_patch  = cfield_rho_d_next_bf_drag.get_buf_check(id);
        sycl::buffer<Tvec>  &rhov_d_patch = cfield_rhov_d_next_bf_drag.get_buf_check(id);

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        q.submit([&, dt](sycl::handler &cgh){
            sycl::accessor acc_dt_rho_patch {dt_rho_patch, cgh, sycl::read_only};
            sycl::accessor acc_dt_rhov_patch {dt_rhov_patch, cgh, sycl::read_only};
            sycl::accessor acc_dt_rhoe_patch {dt_rhoe_patch, cgh, sycl::read_only};

            sycl::accessor acc_rho  {rho_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_rhov {rhov_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_rhoe {rhoe_patch, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, cell_count, "evolve field no drag", [=](u32 id_a) {
                acc_rho[id_a] +=  dt * acc_dt_rho_patch[id_a];
                acc_rhov[id_a] += dt * acc_dt_rhov_patch[id_a];
                acc_rhoe[id_a] += dt * acc_dt_rhoe_patch[id_a];

            });
        });

        q.submit([&, dt, ndust](sycl::handler &cgh){
            sycl::accessor acc_dt_rho_d_patch {dt_rho_d_patch, cgh, sycl::read_only};
            sycl::accessor acc_dt_rhov_d_patch {dt_rhov_d_patch, cgh, sycl::read_only};

            sycl::accessor acc_rho_d  {rho_d_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_rhov_d {rhov_d_patch, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, ndust*cell_count, "dust  evolve field no drag", [=](u32 id_a){
                acc_rho_d[id_a] = dt * acc_dt_rho_d_patch[id_a];
                acc_rhov_d[id_a] = dt * acc_dt_rhov_d_patch[id_a];
            });
        });
    });

    storage.rho_next_no_drag.set(std::move(cfield_rho_next_bf_drag));
    storage.rhov_next_no_drag.set(std::move(cfield_rhov_next_bf_drag));
    storage.rhoe_next_no_drag.set(std::move(cfield_rhoe_next_bf_drag));
    storage.rho_d_next_no_drag.set(std::move(cfield_rho_d_next_bf_drag));
    storage.rhov_d_next_no_drag.set(std::move(cfield_rhov_d_next_bf_drag));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_irk1_drag_integrator(Tscal dt){
    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_new     = storage.rho_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_new     = storage.rhov_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rhoe_new    = storage.rhoe_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rho_d_new   = storage.rho_d_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_d_new   = storage.rhov_d_next_no_drag.get();

    // load layout info
    PatchDataLayout &pdl  = scheduler().pdl;

    const u32 icell_min         = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max         = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho              = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot          = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel           = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d            = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d         = pdl.get_field_idx<Tvec>("rhovel_dust");

    const u32 ndust = solver_config.dust_config.ndust;
    auto alphas_vector = solver_config.drag_config.alphas;
    std::vector<f32> inv_dt_alphas(ndust);
    bool   enable_frictional_heating  = solver_config.drag_config.enable_frictional_heating;
    u32 friction_control = (enable_frictional_heating == false) ? 1 : 0;

    scheduler().for_each_patchdata_nonempty(
        [&, dt, ndust, friction_control](const shamrock::patch::Patch p, shamrock::patch::PatchData & pdat) {
            logger::debug_ln("[AMR enable drag ]", "irk1 drag patch", p.id_patch);

            sycl::queue &q          = shamsys::instance::get_compute_queue();
            u32 id                  = p.id_patch;
            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;
          
            sycl::buffer<Tscal>rho_new_patch     = cfield_rho_new.get_buf_check(id);
            sycl::buffer<Tvec>rhov_new_patch     = cfield_rhov_new.get_buf_check(id);
            sycl::buffer<Tscal>rhoe_new_patch    = cfield_rhoe_new.get_buf_check(id);
            sycl::buffer<Tscal>rho_d_new_patch   = cfield_rho_d_new.get_buf_check(id);
            sycl::buffer<Tvec>rhov_d_new_patch   = cfield_rhov_d_new.get_buf_check(id);

            sycl::buffer<Tscal>rho_old          = pdat.get_field_buf_ref<Tscal>(irho);
            sycl::buffer<Tvec>rhov_old          = pdat.get_field_buf_ref<Tvec>(irhovel);
            sycl::buffer<Tscal>rhoe_old         = pdat.get_field_buf_ref<Tscal>(irhoetot);
            sycl::buffer<Tscal>rho_d_old        = pdat.get_field_buf_ref<Tscal>(irho_d);
            sycl::buffer<Tvec>rhov_d_old        = pdat.get_field_buf_ref<Tvec>(irhovel_d);

            sycl::buffer<f32> alphas_buf {alphas_vector};

            q.submit([&,dt,ndust, friction_control](sycl::handler &cgh) {
                sycl::accessor acc_rho_new_patch{rho_new_patch, cgh, sycl::read_only};
                sycl::accessor acc_rhov_new_patch{rhov_new_patch, cgh, sycl::read_only};
                sycl::accessor acc_rhoe_new_patch{rhoe_new_patch, cgh, sycl::read_only};
                sycl::accessor acc_rho_d_new_patch{rho_d_new_patch, cgh, sycl::read_only};
                sycl::accessor acc_rhov_d_new_patch{rhov_d_new_patch, cgh, sycl::read_only};

                sycl::accessor acc_rho_old {rho_old, cgh, sycl::read_write};
                sycl::accessor acc_rhov_old {rhov_old, cgh, sycl::read_write};
                sycl::accessor acc_rhoe_old {rhoe_old, cgh, sycl::read_write};
                sycl::accessor acc_rho_d_old {rho_d_old, cgh, sycl::read_write};
                sycl::accessor acc_rhov_d_old {rhov_d_old, cgh, sycl::read_write};

                sycl::accessor acc_alphas {alphas_buf, cgh, sycl::read_only};

                shambase::parralel_for(cgh, cell_count, "add_drag [irk1]", [=](u32 id_a){
                        f64_3 tmp_mom_1 = acc_rhov_new_patch[id_a];
                        f64   tmp_rho   = acc_rho_old[id_a];

                        for(u32 i = 0; i < ndust; i++)
                        {
                            const f32 inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                            const f32 dt_alphas     = dt * acc_alphas[i];

                            tmp_mom_1 = tmp_mom_1 + dt_alphas * inv_dt_alphas * acc_rhov_d_new_patch[id_a * ndust + i];
                            tmp_rho   = tmp_rho   + dt_alphas * inv_dt_alphas * acc_rho_d_old[id_a * ndust + i]; 
                        }

                        f64    tmp_inv_rho  = 1.0 / tmp_rho;
                        f64_3  tmp_vel      =  tmp_inv_rho * tmp_mom_1;

                        f64 Eg = 0.0;

                        f64 inv_rho_g      = 1.0 / acc_rho_new_patch[id_a];
                        f64_3 vg_bf        = inv_rho_g * acc_rhov_new_patch[id_a];
                        f64_3 vg_af        = inv_rho_g * acc_rho_old[id_a] * tmp_vel;
                        f64   work_drag    = 0.5 * ( (acc_rho_old[id_a] * tmp_vel[0] - acc_rhov_new_patch[id_a][0]) * (vg_bf[0] + vg_af[0]) + 
                                                     (acc_rho_old[id_a] * tmp_vel[1] - acc_rhov_new_patch[id_a][1]) * (vg_bf[1] + vg_af[1]) +
                                                     (acc_rho_old[id_a] * tmp_vel[2] - acc_rhov_new_patch[id_a][2]) * (vg_bf[2] + vg_af[2])
                                                ) ;
                        
                        f64 dissipation = 0.0;
                        for(u32 i = 0; i < ndust; i++)
                        {
                            const f32 inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                            const f32 dt_alphas     = dt * acc_alphas[i];
                            f64 inv_rho_d           = 1.0 / acc_rho_d_new_patch[id_a];
                            f64_3 vd_bf             = inv_rho_d * acc_rhov_new_patch[id_a * ndust + i] ;  
                            f64_3 vd_af             = inv_rho_d * inv_dt_alphas* (acc_rhov_d_new_patch[id_a * ndust + i] + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);

                            dissipation += 0.5 * dt_alphas * inv_dt_alphas * ( (acc_rho_d_old[id_a * ndust + i] * tmp_vel[0] - acc_rhov_d_new_patch[id_a * ndust + i][0]) * (vd_af[0] + vd_bf[0]) +
                                                    (acc_rho_d_old[id_a * ndust + i] * tmp_vel[1] - acc_rhov_d_new_patch[id_a * ndust + i][1]) * (vd_af[1] + vd_bf[1]) +
                                                    (acc_rho_d_old[id_a * ndust + i] * tmp_vel[2] - acc_rhov_d_new_patch[id_a * ndust + i][2]) * (vd_af[2] + vd_bf[2]))  ;
                        }

                        Eg += acc_rhoe_new_patch[id_a] + (1 - friction_control) * work_drag - friction_control * dissipation;

                        acc_rhov_old[id_a]    = tmp_vel[id_a] * acc_rho_old[id_a];
                        acc_rho_old[id_a]     = acc_rho_new_patch[id_a];
                        acc_rhoe_old[id_a]    = Eg;

                        for(u32 i=0; i < ndust; i++)
                        {
                            const f32 inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                            const f32 dt_alphas     = dt * acc_alphas[i];
                            acc_rhov_d_old[id_a * ndust + i] = inv_dt_alphas * (acc_rhov_d_new_patch[id_a * ndust + i] + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel );
                            acc_rho_d_old[id_a * ndust + i] = acc_rho_d_new_patch[id_a * ndust + i];
                        }
                });
            });
        });

}

template class shammodels::basegodunov::modules::DragIntegrator<f64_3, i64_3>;