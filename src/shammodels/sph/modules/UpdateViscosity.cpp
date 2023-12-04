// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file UpdateViscosity.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief 
 * 
 */

#include "UpdateViscosity.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"
#include "shammodels/sph/math/forces.hpp"
#include <variant>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateViscosity<Tvec, SPHKernel>::update_artificial_viscosity(
    Tscal dt) {

    using Cfg_AV = typename Config::AVConfig;

    using None        = typename Cfg_AV::None;
    using Constant    = typename Cfg_AV::Constant;
    using VaryingMM97 = typename Cfg_AV::VaryingMM97;
    using VaryingCD10 = typename Cfg_AV::VaryingCD10;
    using ConstantDisc = typename Cfg_AV::ConstantDisc;
    if (None *v = std::get_if<None>(&solver_config.artif_viscosity.config)) {
        logger::debug_ln("UpdateViscosity", "skipping artif viscosity update (No viscosity mode)");
    } else if (Constant *v = std::get_if<Constant>(&solver_config.artif_viscosity.config)) {
        logger::debug_ln("UpdateViscosity", "skipping artif viscosity update (Constant mode)");
    } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&solver_config.artif_viscosity.config)) {
        update_artificial_viscosity_mm97(dt, *v);
    } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&solver_config.artif_viscosity.config)) {
        update_artificial_viscosity_cd10(dt, *v);
    } else if (ConstantDisc *v = std::get_if<ConstantDisc>(&solver_config.artif_viscosity.config)) {
        logger::debug_ln("UpdateViscosity", "skipping artif viscosity update (constant AV)");
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateViscosity<Tvec, SPHKernel>::update_artificial_viscosity_mm97(
    Tscal dt, typename Config::AVConfig::VaryingMM97 cfg) {
    StackEntry stack_loc{};
    logger::debug_ln("UpdateViscosity", "Updating alpha viscosity (Morris & Monaghan 1997)");

    using namespace shamrock::patch;
    PatchDataLayout &pdl  = scheduler().pdl;
    const u32 ialpha_AV   = pdl.get_field_idx<Tscal>("alpha_AV");
    const u32 idivv       = pdl.get_field_idx<Tscal>("divv");
    const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");
    const u32 ihpart      = pdl.get_field_idx<Tscal>("hpart");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::buffer<Tscal> &buf_divv     = pdat.get_field_buf_ref<Tscal>(idivv);
        sycl::buffer<Tscal> &buf_cs       = pdat.get_field_buf_ref<Tscal>(isoundspeed);
        sycl::buffer<Tscal> &buf_h        = pdat.get_field_buf_ref<Tscal>(ihpart);
        sycl::buffer<Tscal> &buf_alpha_AV = pdat.get_field_buf_ref<Tscal>(ialpha_AV);

        u32 obj_cnt = pdat.get_obj_cnt();

        shamsys::instance::get_compute_queue().submit([&, dt](sycl::handler &cgh) {
            sycl::accessor divv{buf_divv, cgh, sycl::read_only};
            sycl::accessor cs{buf_cs, cgh, sycl::read_only};
            sycl::accessor h{buf_h, cgh, sycl::read_only};
            sycl::accessor alpha_AV{buf_alpha_AV, cgh, sycl::read_write};

            Tscal sigma_decay = cfg.sigma_decay;
            Tscal alpha_min   = cfg.alpha_min;
            Tscal alpha_max   = cfg.alpha_max;

            cgh.parallel_for(sycl::range<1>{obj_cnt}, [=](sycl::item<1> item) {
                using namespace shambase::sycl_utils;

                Tscal cs_a    = cs[item];
                Tscal h_a     = h[item];
                Tscal alpha_a = alpha_AV[item];
                Tscal divv_a  = divv[item];

                Tscal vsig            = cs_a;
                Tscal inv_tau_a       = vsig * sigma_decay / h_a;
                Tscal fact_t          = dt * inv_tau_a;
                Tscal euler_impl_fact = 1 / (1 + fact_t);

                Tscal source = g_sycl_max<Tscal>(0., -divv_a);

                Tscal new_alpha = (alpha_a + source * dt + fact_t * alpha_min) * euler_impl_fact;

                alpha_AV[item] = g_sycl_min(alpha_max, new_alpha);
            });
        });
    });
}


template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateViscosity<Tvec, SPHKernel>::update_artificial_viscosity_cd10(
    Tscal dt, typename Config::AVConfig::VaryingCD10 cfg) {

    StackEntry stack_loc{};
    logger::debug_ln("UpdateViscosity", "Updating alpha viscosity (Cullen & Dehnen 2010)");

    using namespace shamrock::patch;
    PatchDataLayout &pdl  = scheduler().pdl;
    const u32 ialpha_AV   = pdl.get_field_idx<Tscal>("alpha_AV");
    const u32 idivv       = pdl.get_field_idx<Tscal>("divv");
    const u32 idtdivv       = pdl.get_field_idx<Tscal>("dtdivv");
    const u32 icurlv       = pdl.get_field_idx<Tvec>("curlv");
    const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");
    const u32 ihpart      = pdl.get_field_idx<Tscal>("hpart");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::buffer<Tscal> &buf_divv     = pdat.get_field_buf_ref<Tscal>(idivv);
        sycl::buffer<Tscal> &buf_dtdivv     = pdat.get_field_buf_ref<Tscal>(idtdivv);
        sycl::buffer<Tvec> &buf_curlv     = pdat.get_field_buf_ref<Tvec>(icurlv);
        sycl::buffer<Tscal> &buf_cs       = pdat.get_field_buf_ref<Tscal>(isoundspeed);
        sycl::buffer<Tscal> &buf_h        = pdat.get_field_buf_ref<Tscal>(ihpart);
        sycl::buffer<Tscal> &buf_alpha_AV = pdat.get_field_buf_ref<Tscal>(ialpha_AV);

        u32 obj_cnt = pdat.get_obj_cnt();

        shamsys::instance::get_compute_queue().submit([&, dt](sycl::handler &cgh) {
            sycl::accessor divv{buf_divv, cgh, sycl::read_only};
            sycl::accessor curlv{buf_curlv, cgh, sycl::read_only};
            sycl::accessor dtdivv{buf_dtdivv, cgh, sycl::read_only};
            sycl::accessor cs{buf_cs, cgh, sycl::read_only};
            sycl::accessor h{buf_h, cgh, sycl::read_only};
            sycl::accessor alpha_AV{buf_alpha_AV, cgh, sycl::read_write};

            Tscal sigma_decay = cfg.sigma_decay;
            Tscal alpha_min   = cfg.alpha_min;
            Tscal alpha_max   = cfg.alpha_max;

            cgh.parallel_for(sycl::range<1>{obj_cnt}, [=](sycl::item<1> item) {
                using namespace shambase::sycl_utils;

                Tscal cs_a    = cs[item];
                Tscal h_a     = h[item];
                Tscal alpha_a = alpha_AV[item];
                Tscal divv_a  = divv[item];
                Tvec curlv_a  = curlv[item];
                Tscal dtdivv_a  = dtdivv[item];

                

                Tscal vsig            = cs_a;
                Tscal inv_tau_a       = vsig * sigma_decay / h_a;
                Tscal fact_t          = dt * inv_tau_a;
                Tscal euler_impl_fact = 1 / (1 + fact_t);



                //Tscal div_corec = g_sycl_max<Tscal>(-divv_a, 0);
                //Tscal divv_a_sq = div_corec*div_corec;
                ////Tscal divv_a_sq_corec = g_sycl_max(-divv_a, 0);
                //Tscal curlv_a_sq = sycl::dot(curlv_a,curlv_a);
                //Tscal denom = (curlv_a_sq + divv_a_sq);
                //Tscal balsara_corec = (denom <= 0) ? 1 : divv_a_sq / (curlv_a_sq + divv_a_sq);
                
                auto xi_lim = [](Tscal divv, Tvec curlv){
                    auto fac = sham::max(-divv, Tscal{0});
                    fac *= fac;
                    auto traceS = sycl::dot(curlv,curlv);
                    if (fac + traceS > 1e-12) {
                        return fac/(fac + traceS);
                    }
                    return Tscal{1};
                };
                Tscal balsara_corec = xi_lim(divv_a,curlv_a);


                Tscal A_a = balsara_corec*g_sycl_max<Tscal>(-dtdivv_a, 0);




                Tscal alpha_loc_a = g_sycl_min(
                        (cs_a > 0) ? 10*h_a*h_a*A_a/(cs_a*cs_a) : alpha_min
                    ,alpha_max);

                //implicit euler
                Tscal new_alpha = (alpha_a + alpha_loc_a * fact_t) * euler_impl_fact;

                if(alpha_loc_a > alpha_a){
                    new_alpha = alpha_loc_a;
                }

                alpha_AV[item] = new_alpha;
            });
        });
    });
}

using namespace shammath;
template class shammodels::sph::modules::UpdateViscosity<f64_3, M4>;
template class shammodels::sph::modules::UpdateViscosity<f64_3, M6>;