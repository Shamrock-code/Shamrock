// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConservativeCheck.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ConservativeCheck.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ConservativeCheck<Tvec, SPHKernel>::check_conservation() {

    StackEntry stack_loc{};

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;
    using Sink = SinkParticle<Tvec>;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");

    bool has_B_field     = solver_config.has_field_B_on_rho();
    const u32 iB_on_rho  = (has_B_field) ? pdl.get_field_idx<Tvec>("B/rho") : -1;
    const u32 idB_on_rho = (has_B_field) ? pdl.get_field_idx<Tvec>("dB/rho") : -1;
    const u32 idrho_dt   = (has_B_field) ? pdl.get_field_idx<Tscal>("drho/dt") : -1;

    std::string cv_checks = "conservation infos :\n";

    ///////////////////////////////////
    // momentum check :
    ///////////////////////////////////
    Tvec tmpp{0, 0, 0};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(ivxyz);
        tmpp += field.compute_sum();
    });
    Tvec sum_p = gpart_mass * shamalgs::collective::allreduce_sum(tmpp);

    if (shamcomm::world_rank() == 0) {
        if (!storage.sinks.is_empty()) {
            std::vector<Sink> &sink_parts = storage.sinks.get();
            for (Sink &s : sink_parts) {
                sum_p += s.mass * s.velocity;
            }
        }
        cv_checks += shambase::format("    sum v = {}\n", sum_p);
    }

    ///////////////////////////////////
    // force sum check :
    ///////////////////////////////////
    Tvec tmpa{0, 0, 0};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(iaxyz);
        tmpa += field.compute_sum();
    });
    Tvec sum_a = gpart_mass * shamalgs::collective::allreduce_sum(tmpa);

    if (shamcomm::world_rank() == 0) {
        if (!storage.sinks.is_empty()) {
            std::vector<Sink> &sink_parts = storage.sinks.get();
            for (Sink &s : sink_parts) {
                sum_a += s.mass * (s.sph_acceleration + s.ext_acceleration);
            }
        }
        cv_checks += shambase::format("    sum a = {}\n", sum_a);
    }

    ///////////////////////////////////
    // energy check :
    ///////////////////////////////////
    Tscal tmpe{0};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataField<Tscal> &field_u = pdat.get_field<Tscal>(iuint);
        PatchDataField<Tvec> &field_v  = pdat.get_field<Tvec>(ivxyz);
        tmpe += field_u.compute_sum() + 0.5 * field_v.compute_dot_sum();
    });
    Tscal sum_e = gpart_mass * shamalgs::collective::allreduce_sum(tmpe);

    if (shamcomm::world_rank() == 0) {
        cv_checks += shambase::format("    sum e = {}\n", sum_e);
    }

    Tscal pmass  = gpart_mass;
    Tscal tmp_de = 0;
    scheduler().for_each_patchdata_nonempty([&, pmass](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataField<Tvec> &field_v      = pdat.get_field<Tvec>(ivxyz);
        PatchDataField<Tscal> &field_du    = pdat.get_field<Tscal>(iduint);
        PatchDataField<Tvec> &field_a      = pdat.get_field<Tvec>(iaxyz);
        PatchDataField<Tscal> &field_hpart = pdat.get_field<Tscal>(ihpart);

        sham::DeviceBuffer<Tscal> temp_de(pdat.get_obj_cnt(), dev_sched);

        Tscal const mu_0 = solver_config.get_constant_mu_0();

        sham::kernel_call(
            q,
            sham::MultiRef{field_du.get_buf(), field_v.get_buf(), field_a.get_buf()},
            sham::MultiRef{temp_de},
            pdat.get_obj_cnt(),
            [=](u32 item, const Tscal *du, const Tvec *v, const Tvec *a, Tscal *de) {
                de[item] = pmass * (sycl::dot(v[item], a[item]) + du[item]);
            });

        if (has_B_field) {
            PatchDataField<Tvec> &field_B_on_rho  = pdat.get_field<Tvec>(iB_on_rho);
            PatchDataField<Tvec> &field_dB_on_rho = pdat.get_field<Tvec>(idB_on_rho);
            PatchDataField<Tscal> &field_drho_dt  = pdat.get_field<Tscal>(idrho_dt);

            sham::kernel_call(
                q,
                sham::MultiRef{
                    field_hpart.get_buf(),
                    field_B_on_rho.get_buf(),
                    field_dB_on_rho.get_buf(),
                    field_drho_dt.get_buf()},
                sham::MultiRef{temp_de},
                pdat.get_obj_cnt(),
                [=](u32 item,
                    const Tscal *hpart,
                    const Tvec *B_on_rho,
                    const Tvec *dB_on_rho,
                    const Tscal *drho_dt,
                    Tscal *de) {
                    using namespace shamrock::sph;
                    Tscal h      = hpart[item];
                    Tscal term_B = 0.;

                    Tvec B_on_rho_a  = B_on_rho[item];
                    Tvec B           = B_on_rho_a * shamrock::sph::rho_h(pmass, h, Kernel::hfactd);
                    Tvec dB_on_rho_a = dB_on_rho[item];
                    Tscal drho       = drho_dt[item];
                    term_B           = 0.5 * (1. / mu_0) * sycl::dot(B_on_rho_a, B_on_rho_a) * drho
                             + (1. / mu_0) * sycl::dot(B, dB_on_rho_a);

                    de[item] += pmass * term_B;
                });
        }

        Tscal de_p = shamalgs::primitives::sum(dev_sched, temp_de, 0, pdat.get_obj_cnt());
        tmp_de += de_p;
    });

    Tscal de = shamalgs::collective::allreduce_sum(tmp_de);

    if (shamcomm::world_rank() == 0) {
        cv_checks += shambase::format("    sum de = {}", de);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sph::Model", cv_checks);
    }
}

using namespace shammath;
template class shammodels::sph::modules::ConservativeCheck<f64_3, M4>;
template class shammodels::sph::modules::ConservativeCheck<f64_3, M6>;
template class shammodels::sph::modules::ConservativeCheck<f64_3, M8>;

template class shammodels::sph::modules::ConservativeCheck<f64_3, C2>;
template class shammodels::sph::modules::ConservativeCheck<f64_3, C4>;
template class shammodels::sph::modules::ConservativeCheck<f64_3, C6>;
