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
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Implementation of GSPH solver configuration methods
 */

#include "shammodels/gsph/SolverConfig.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/config/FieldNames.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_layout(
    shamrock::patch::PatchDataLayerLayout &pdl) {

    // Position
    pdl.add_field<Tvec>(names::common::xyz, 1);

    // Velocity
    pdl.add_field<Tvec>(names::newtonian::vxyz, 1);

    // Acceleration
    pdl.add_field<Tvec>(names::newtonian::axyz, 1);

    // Smoothing length
    pdl.add_field<Tscal>(names::common::hpart, 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        pdl.add_field<Tscal>(names::newtonian::uint, 1);
        pdl.add_field<Tscal>(names::newtonian::duint, 1);
    }

    // Thermodynamic fields - stored in patchdata for persistence across restarts
    // These are computed during EOS step and copied to patchdata
    pdl.add_field<Tscal>(names::newtonian::density, 1);
    pdl.add_field<Tscal>(names::newtonian::pressure, 1);
    pdl.add_field<Tscal>(names::newtonian::soundspeed, 1);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_ghost_layout(
    shamrock::patch::PatchDataLayerLayout &ghost_layout) {

    // Velocity (needed for Riemann solver)
    ghost_layout.add_field<Tvec>(names::newtonian::vxyz, 1);

    // Smoothing length
    ghost_layout.add_field<Tscal>(names::common::hpart, 1);

    // Omega (grad-h correction)
    ghost_layout.add_field<Tscal>(names::newtonian::omega, 1);

    // Density (computed via SPH summation)
    ghost_layout.add_field<Tscal>(names::newtonian::density, 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        ghost_layout.add_field<Tscal>(names::newtonian::uint, 1);
    }
}

// Explicit template instantiations
using namespace shammath;
template class shammodels::gsph::SolverConfig<f64_3, M4>;
template class shammodels::gsph::SolverConfig<f64_3, M6>;
template class shammodels::gsph::SolverConfig<f64_3, M8>;
template class shammodels::gsph::SolverConfig<f64_3, C2>;
template class shammodels::gsph::SolverConfig<f64_3, C4>;
template class shammodels::gsph::SolverConfig<f64_3, C6>;
