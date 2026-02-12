// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverConfig.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shammodels/sph/SolverConfig.hpp"

namespace shammodels::sph {

    template<class Tvec, template<class> class SPHKernel>
    void SolverConfig<Tvec, SPHKernel>::set_layout(shamrock::patch::PatchDataLayerLayout &pdl) {
        pdl.add_field<Tvec>("xyz", 1);
        pdl.add_field<Tvec>("vxyz", 1);
        pdl.add_field<Tvec>("axyz", 1);
        pdl.add_field<Tvec>("axyz_ext", 1);
        pdl.add_field<Tscal>("hpart", 1);

        if (track_particles_id) {
            pdl.add_field<u64>("part_id", 1);
        }

        if (has_field_uint()) {
            pdl.add_field<Tscal>("uint", 1);
            pdl.add_field<Tscal>("duint", 1);
        }

        if (has_field_alphaAV()) {
            pdl.add_field<Tscal>("alpha_AV", 1);
        }

        if (has_field_divv()) {
            pdl.add_field<Tscal>("divv", 1);
        }

        if (has_field_dtdivv()) {
            pdl.add_field<Tscal>("dtdivv", 1);
        }

        if (has_field_curlv()) {
            pdl.add_field<Tvec>("curlv", 1);
        }

        if (has_field_soundspeed()) {

            // this should not be needed idealy, but we need the pressure on the ghosts and
            // we don't want to communicate it as it can be recomputed from the other fields
            // hence we copy the soundspeed at the end of the step to a field in the patchdata
            pdl.add_field<Tscal>("soundspeed", 1);
        }

        if (has_field_B_on_rho()) {

            pdl.add_field<Tvec>("B/rho", 1);
            pdl.add_field<Tvec>("dB/rho", 1);
            pdl.add_field<Tscal>("drho/dt", 1);
        }

        if (has_field_psi_on_ch()) {
            pdl.add_field<Tscal>("psi/ch", 1);
            pdl.add_field<Tscal>("dpsi/ch", 1);
        }
        if (has_field_divB()) {
            pdl.add_field<Tscal>("divB", 1);
        }

        if (has_field_curlB()) {
            pdl.add_field<Tvec>("curlB", 1);
        }

        if (dust_config.has_epsilon_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            pdl.add_field<Tscal>("epsilon", ndust);
            pdl.add_field<Tscal>("dtepsilon", ndust);
        }

        if (dust_config.has_deltav_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            pdl.add_field<Tvec>("deltav", ndust);
            pdl.add_field<Tvec>("dtdeltav", ndust);
        }
        if (do_MHD_debug()) {
            pdl.add_field<Tvec>("gas_pressure", 1);
            pdl.add_field<Tvec>("mag_pressure", 1);
            pdl.add_field<Tvec>("mag_tension", 1);
            pdl.add_field<Tvec>("tensile_corr", 1);

            pdl.add_field<Tscal>("psi_propag", 1);
            pdl.add_field<Tscal>("psi_diff", 1);
            pdl.add_field<Tscal>("psi_cons", 1);
            pdl.add_field<Tscal>("u_mhd", 1);
        }
        if (true) { // has_field_disable_mask()
            pdl.add_field<u32>("ghost_mask", 1);
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void SolverConfig<Tvec, SPHKernel>::set_ghost_layout(
        shamrock::patch::PatchDataLayerLayout &ghost_layout) {

        ghost_layout.add_field<Tscal>("hpart", 1);
        ghost_layout.add_field<Tscal>("uint", 1);
        ghost_layout.add_field<Tvec>("vxyz", 1);

        if (has_axyz_in_ghost()) {
            ghost_layout.add_field<Tvec>("axyz", 1);
        }
        ghost_layout.add_field<Tscal>("omega", 1);

        if (ghost_has_soundspeed()) {
            ghost_layout.add_field<Tscal>("soundspeed", 1);
        }

        if (has_field_B_on_rho()) {
            ghost_layout.add_field<Tvec>("B/rho", 1);
        }

        if (has_field_psi_on_ch()) {
            ghost_layout.add_field<Tscal>("psi/ch", 1);
        }

        if (has_field_curlB()) {
            ghost_layout.add_field<Tvec>("curlB", 1);
        }

        if (dust_config.has_epsilon_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            ghost_layout.add_field<Tscal>("epsilon", ndust);
        }

        if (dust_config.has_deltav_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            ghost_layout.add_field<Tvec>("deltav", ndust);
        }
    }

}; // namespace shammodels::sph

using namespace shammath;

template class shammodels::sph::SolverConfig<f64_3, M4>;
template class shammodels::sph::SolverConfig<f64_3, M6>;
template class shammodels::sph::SolverConfig<f64_3, M8>;

template class shammodels::sph::SolverConfig<f64_3, C2>;
template class shammodels::sph::SolverConfig<f64_3, C4>;
template class shammodels::sph::SolverConfig<f64_3, C6>;
