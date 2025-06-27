// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ParticleInjection.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ParticleInjection.hpp"
#include "shamalgs/random.hpp"
#include "shammath/sphkernels.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include <vector>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ParticleInjection<Tvec, SPHKernel>::inject_particles() {

    Tscal inject_mass_rate = 1e-4;
    Tscal part_mass        = solver_config.gpart_mass;
    Tscal part_inject_rate = inject_mass_rate / part_mass;

    Tscal part_to_inject = part_inject_rate * solver_config.get_dt_sph();

    Tvec pos_inject = {1.0, 0.3, 0.0};
    Tscal r_inject  = 0.4;

    using namespace shamrock::patch;
    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    auto has_pdat = [&]() {
        bool ret = false;
        sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
            ret = true;
        });
        return ret;
    };

    std::vector<Tvec> pos_gen;
    std::vector<Tvec> vel_gen;

    static std::mt19937 eng(0);

    for (u32 i = 0; i < part_to_inject; i++) {
        Tvec bound_inject_low  = Tvec{-r_inject, -r_inject, -r_inject} + pos_inject;
        Tvec bound_inject_high = Tvec{r_inject, r_inject, r_inject} + pos_inject;
        auto r = shamalgs::random::mock_value<Tvec>(eng, bound_inject_low, bound_inject_high);
        auto v = shamalgs::random::mock_value<Tvec>(eng, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

        pos_gen.push_back(r);
        vel_gen.push_back(v);
    }

    PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

    auto correct_smoothing_lengths = [&](PatchData &pdat, PatchData &to_insert) {
        auto h_field_idx = sched.pdl.get_field_idx<Tscal>("hpart");

        PatchDataField<Tscal> &h_field        = pdat.get_field<Tscal>(h_field_idx);
        PatchDataField<Tscal> &h_field_insert = to_insert.get_field<Tscal>(h_field_idx);

        auto h_min = h_field.compute_min();
        h_field_insert.override(h_min);
    };

    sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
        shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

        std::vector<Tvec> pos;
        std::vector<Tvec> vel;

        for (u64 i = 0; i < pos_gen.size(); i++) {
            Tvec r = pos_gen[i];
            if (patch_coord.contain_pos(r)) {
                pos.push_back(pos_gen[i]);
                vel.push_back(vel_gen[i]);
            }
        }

        PatchData tmp(sched.pdl);

        if (!pos.empty()) {
            tmp.resize(pos.size());
            tmp.fields_raz();

            {
                u32 len                 = pos.size();
                PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(pos.data(), len);
                f.override(buf, len);
            }

            {
                u32 len = pos.size();
                PatchDataField<Tvec> &f
                    = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
                sycl::buffer<Tvec> buf(vel.data(), len);
                f.override(buf, len);
            }

            correct_smoothing_lengths(pdat, tmp);

            pdat.insert_elements(tmp);
        }
    });

    sched.check_patchdata_locality_corectness();
}

using namespace shammath;
template class shammodels::sph::modules::ParticleInjection<f64_3, M4>;
template class shammodels::sph::modules::ParticleInjection<f64_3, M6>;
template class shammodels::sph::modules::ParticleInjection<f64_3, M8>;
