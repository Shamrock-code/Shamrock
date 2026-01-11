// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

    // Common fields (all physics modes)
    pdl.add_field<Tvec>(names::common::xyz, 1);
    pdl.add_field<Tscal>(names::common::hpart, 1);

    // Newtonian physics fields
    pdl.add_field<Tvec>(names::newtonian::vxyz, 1);
    pdl.add_field<Tvec>(names::newtonian::axyz, 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        pdl.add_field<Tscal>(names::newtonian::uint, 1);
        pdl.add_field<Tscal>(names::newtonian::duint, 1);
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_ghost_layout(
    shamrock::patch::PatchDataLayerLayout &ghost_layout) {

    // Newtonian physics ghost fields
    ghost_layout.add_field<Tvec>(names::newtonian::vxyz, 1);
    ghost_layout.add_field<Tscal>(names::common::hpart, 1);
    ghost_layout.add_field<Tscal>(names::newtonian::omega, 1);
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
