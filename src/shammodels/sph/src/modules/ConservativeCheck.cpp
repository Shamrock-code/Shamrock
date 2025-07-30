// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConservativeCheck.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/ConservativeCheck.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ConservativeCheck<Tvec, SPHKernel>::check_conservation() {

    StackEntry stack_loc{};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;
    using Sink = SinkParticle<Tvec>;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");

    std::string cv_checks = "conservation infos :\n";

    ///////////////////////////////////
    // momentum check :
    ///////////////////////////////////
    Tvec tmpp{0, 0, 0};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
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
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
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
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
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
    scheduler().for_each_patchdata_nonempty([&, pmass](Patch cur_p, PatchData &pdat) {
        PatchDataField<Tscal> &field_u  = pdat.get_field<Tscal>(iuint);
        PatchDataField<Tvec> &field_v   = pdat.get_field<Tvec>(ivxyz);
        PatchDataField<Tscal> &field_du = pdat.get_field<Tscal>(iduint);
        PatchDataField<Tvec> &field_a   = pdat.get_field<Tvec>(iaxyz);

        sycl::buffer<Tscal> temp_de(pdat.get_obj_cnt());

        sham::EventList depends_list;
        auto u  = field_u.get_buf().get_read_access(depends_list);
        auto du = field_du.get_buf().get_read_access(depends_list);
        auto v  = field_v.get_buf().get_read_access(depends_list);
        auto a  = field_a.get_buf().get_read_access(depends_list);

        auto e = q.submit(depends_list, [&, pmass](sycl::handler &cgh) {
            sycl::accessor de{temp_de, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                de[item] = pmass * (sycl::dot(v[item], a[item]) + du[item]);
            });
        });

        field_u.get_buf().complete_event_state(e);
        field_du.get_buf().complete_event_state(e);
        field_v.get_buf().complete_event_state(e);
        field_a.get_buf().complete_event_state(e);

        Tscal de_p = shamalgs::reduction::sum(
            shamsys::instance::get_compute_queue(), temp_de, 0, pdat.get_obj_cnt());
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
