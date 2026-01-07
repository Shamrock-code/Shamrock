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
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Implementation of GSPH solver configuration methods
 */

#include "shammodels/gsph/SolverConfig.hpp"
#include "shammath/sphkernels.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_layout(
    shamrock::patch::PatchDataLayerLayout &pdl) {

    // Position
    pdl.add_field<Tvec>("xyz", 1);

    // Velocity
    pdl.add_field<Tvec>("vxyz", 1);

    // Acceleration
    pdl.add_field<Tvec>("axyz", 1);

    // Smoothing length
    pdl.add_field<Tscal>("hpart", 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        pdl.add_field<Tscal>("uint", 1);
        pdl.add_field<Tscal>("duint", 1);
    }

    // Density field for direct output (computed by SPH summation)
    pdl.add_field<Tscal>("rho", 1);

    // Physics-specific fields are added via PhysicsMode::extend_layout()
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_ghost_layout(
    shamrock::patch::PatchDataLayerLayout &ghost_layout) {

    // Velocity (needed for Riemann solver)
    ghost_layout.add_field<Tvec>("vxyz", 1);

    // Smoothing length
    ghost_layout.add_field<Tscal>("hpart", 1);

    // Omega (grad-h correction)
    ghost_layout.add_field<Tscal>("omega", 1);

    // Density (computed via SPH summation)
    ghost_layout.add_field<Tscal>("density", 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        ghost_layout.add_field<Tscal>("uint", 1);
    }

    // Physics-specific ghost fields are added via PhysicsMode::extend_ghost_layout()
}

// Explicit template instantiations
using namespace shammath;
template class shammodels::gsph::SolverConfig<f64_3, M4>;
template class shammodels::gsph::SolverConfig<f64_3, M6>;
template class shammodels::gsph::SolverConfig<f64_3, M8>;
template class shammodels::gsph::SolverConfig<f64_3, C2>;
template class shammodels::gsph::SolverConfig<f64_3, C4>;
template class shammodels::gsph::SolverConfig<f64_3, C6>;
template class shammodels::gsph::SolverConfig<f64_3, TGauss3>;
