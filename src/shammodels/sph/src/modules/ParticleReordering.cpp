// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ParticleReordering.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ParticleReordering.hpp"
#include "shammath/sphkernels.hpp"

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::ParticleReordering<Tvec, Tmorton, SPHKernel>::reorder_particles() {

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchCoordTransform<Tvec> transf
        = scheduler().get_sim_box().template get_patch_transform<Tvec>();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        u32 obj_count = pdat.get_obj_cnt();

        if (obj_count > 0) {
            PatchDataField<Tvec> &pos_field = pdat.get_field<Tvec>(0);

            shammath::CoordRange<Tvec> box = transf.to_obj_coord(cur_p);

            shamrock::tree::TreeMortonCodes<Tmorton> builder;
            builder.build(
                shamsys::instance::get_compute_scheduler_ptr(),
                box,
                obj_count,
                pos_field.get_buf());

            pdat.index_remap(shambase::get_check_ref(builder.buf_particle_index_map), obj_count);
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::ParticleReordering<f64_3, u32, M4>;
template class shammodels::sph::modules::ParticleReordering<f64_3, u32, M6>;
template class shammodels::sph::modules::ParticleReordering<f64_3, u32, M8>;

template class shammodels::sph::modules::ParticleReordering<f64_3, u32, C2>;
template class shammodels::sph::modules::ParticleReordering<f64_3, u32, C4>;
template class shammodels::sph::modules::ParticleReordering<f64_3, u32, C6>;
