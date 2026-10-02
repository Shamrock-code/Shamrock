// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GeneratorLatticeFCC.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief SPH setup generator placing particles on a FCC lattice
 *
 */

#include "shammath/crystalLattice.hpp"
#include "shammodels/sph/modules/setup/GeneratorLattice.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, bool Discontinuous = true>
    using GeneratorLatticeFCC = GeneratorLattice<Tvec, shammath::LatticeFCC<Tvec>, Discontinuous>;

} // namespace shammodels::sph::modules
