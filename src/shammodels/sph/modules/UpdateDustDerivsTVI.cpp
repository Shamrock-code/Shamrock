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
 * @brief 
 * 
 */

#include "UpdateDustDerivsTVI.hpp"
#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"


template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDustDerivsTVI<Tvec, SPHKernel>::update_dust_derivs() {

    
    using DustCFG     = typename Config::DustConfig;
    using DustCFGNone = typename DustCFG::None;
    using DustCFGMonofluidTVI = typename DustCFG::DustMonofluidTvi;

    if (DustCFGNone *dust_config = std::get_if<DustCFGNone>(&solver_config.dust_config.config)) {
        logger::debug_ln("SPHDust", "skip derivs there if no dust :'( ");
    }else if (DustCFGMonofluidTVI *dust_config = std::get_if<DustCFGMonofluidTVI>(&solver_config.dust_config.config)) {
        update_dust_derivs_tvi(*dust_config);
    }else{
        shambase::throw_unimplemented();
    }

    shamrock::ComputeField<Tscal> & soundspeed = storage.soundspeed.get();

    //Tscal rho_grain = solver_config.

}

/**
 * @brief Get the Kj (Hutchison 2O18)
 *
 * \f$ K_j = \frac{\rho_g \rho_d c_s}{\rho_{\rm eff} s_j} \f$
 *
 * @tparam Tscal 
 * @param rhog 
 * @param rhod 
 * @param cs 
 * @param rhoeff 
 * @param sj 
 * @return Tscal 
 */
template<class Tscal>
Tscal get_Kj(Tscal rhog, Tscal rhod, Tscal cs, Tscal rhoeff, Tscal sj){
    return rhog * rhod * cs / (rhoeff * sj);
}

template<class Tscal>
Tscal get_tj(Tscal rhog, Tscal rhod, Tscal cs, Tscal rhoeff, Tscal sj){
    return (rhog + rhod)/get_Kj(rhog, rhod, cs, rhoeff, sj);
}

template<class Tscal>
Tscal get_Tsj(Tscal eps_j, Tscal eps, Tscal tj){
    return eps_j * (1 - eps)*tj;
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDustDerivsTVI<Tvec, SPHKernel>::update_dust_derivs_tvi(DustCFGMonofluidTVI config) {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;
    using namespace shambase::constants;

    shamrock::ComputeField<Tscal> & soundspeed = storage.soundspeed.get();

    // which gamma in isothermal
    Tscal gamma = 1;

    Tscal rho_grain = config.rho_grain;
    Tscal rho_eff = sycl::sqrt(pi<Tscal> * gamma /8);



    PatchDataLayout &pdl = scheduler().pdl;
    const u32 iSdust   = pdl.get_field_idx<Tscal>("Sdust");
    const u32 isSdust   = pdl.get_field_idx<Tscal>("dSdust");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

    u32 iSdust_interf  = ghost_layout.get_field_idx<Tscal>("Sdust"); 
    u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf    = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();


    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;


        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_vxyz      = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tscal> &buf_hpart    = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega    = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_pressure = storage.pressure.get().get_buf_check(cur_p.id_patch);


        sycl::buffer<Tscal> &buf_dS    = pdat.get_field_buf_ref<Tscal>(isSdust);
        sycl::buffer<Tscal> &buf_S    = mpdat.get_field_buf_ref<Tscal>(iSdust_interf);



        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal gamma = gamma;
            const Tscal rhoeff = rho_eff;

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
            sycl::accessor omega{buf_omega, cgh, sycl::read_only};
            sycl::accessor pressure{buf_pressure, cgh, sycl::read_only};
            sycl::accessor cs{
                storage.soundspeed.get().get_buf_check(cur_p.id_patch), cgh, sycl::read_only};

            sycl::accessor dS {buf_dS, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor S {buf_S, cgh, sycl::write_only, sycl::no_init};


            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute dust S derivative", [=](u64 gid) {
                u32 id_a = (u32)gid;

                constexpr Tscal m1half = -0.5;

                using namespace shamrock::sph;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal omega_a   = omega[id_a];
                Tscal cs_a = cs[id_a];
                Tscal S_a = S[id_a];



                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);

                Tscal dS_sum = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {

                    Tvec dr    = xyz_a - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    Tscal rab       = sycl::sqrt(rab2);
                    Tvec r_ab_unit = dr / rab;
                    Tscal inv_rab = 1/rab;

                    if (rab < 1e-9) {
                        r_ab_unit = {0, 0, 0};
                        inv_rab = 0;
                    }

                    Tvec vxyz_b     = vxyz[id_b];
                    Tvec v_ab       = vxyz_a - vxyz_b;
                    Tscal S_b = S[id_b];
                    Tscal P_b   = pressure[id_b];

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);

                    Tscal Tsja = 0;
                    Tscal Tsjb = 0;


                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);



                    dS_sum += m1half * (pmass * S_b / rho_b) * (
                        Tsja/rho_a + Tsjb/rho_b
                    )* (P_a - P_b)* 0.5*(Fab_a + Fab_b)*inv_rab +
                    (S_a / (2*rho_a*omega_a)) * pmass *sycl::dot(v_ab, r_ab_unit)*Fab_a;

                });

                
                dS[id_a] = dS_sum;


            });

        });


    });

}




using namespace shammath;
template class shammodels::sph::modules::UpdateDustDerivsTVI<f64_3, M4>;
template class shammodels::sph::modules::UpdateDustDerivsTVI<f64_3, M6>;
template class shammodels::sph::modules::UpdateDustDerivsTVI<f64_3, M8>;