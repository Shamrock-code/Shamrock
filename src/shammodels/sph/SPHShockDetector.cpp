// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "SPHShockDetector.hpp"
#include "shamrock/sph/kernels.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::SPHShockDetector<Tvec, SPHKernel>::update_artificial_viscosity(
    Tscal dt, typename Config::AVConfig::Variant cfg) {

    using Cfg_AV = typename Config::AVConfig;

    using None        = typename Cfg_AV::None;
    using Constant    = typename Cfg_AV::Constant;
    using VaryingMM97 = typename Cfg_AV::VaryingMM97;
    using VaryingCD10 = typename Cfg_AV::VaryingCD10;
    if (None *v = std::get_if<None>(&cfg)) {
        logger::debug_ln("SPHSolver", "skipping artif viscosity update (No viscosity mode)");
    } else if (Constant *v = std::get_if<Constant>(&cfg)) {
        logger::debug_ln("SPHSolver", "skipping artif viscosity update (Constant mode)");
    } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&cfg)) {
        update_artificial_viscosity_mm97(dt, *v);
    } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&cfg)) {
        update_artificial_viscosity_cd10(dt, *v);
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::SPHShockDetector<Tvec, SPHKernel>::update_artificial_viscosity_mm97(
    Tscal dt, typename Config::AVConfig::VaryingMM97 cfg) {
    StackEntry stack_loc{};
    logger::info_ln("SPHShockDetector", "Updating alpha viscosity (Morris & Monaghan 1997)");

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
void shammodels::SPHShockDetector<Tvec, SPHKernel>::update_artificial_viscosity_cd10(
    Tscal dt, typename Config::AVConfig::VaryingCD10 cfg) {

    shambase::throw_unimplemented();
}

using namespace shamrock::sph::kernels;
template class shammodels::SPHShockDetector<f64_3, M4>;
template class shammodels::SPHShockDetector<f64_3, M6>;