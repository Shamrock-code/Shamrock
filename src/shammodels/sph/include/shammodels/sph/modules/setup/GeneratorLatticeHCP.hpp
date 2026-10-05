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
 * @file GeneratorLatticeHCP.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief SPH setup generator placing particles on a HCP lattice
 *
 */

#include "shammath/crystalLattice.hpp"
#include "shammodels/sph/modules/setup/GeneratorLattice.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, bool discontinuous = true>
    using GeneratorLatticeHCP = GeneratorLattice<Tvec, shammath::LatticeHCP<Tvec>, discontinuous>;

} // namespace shammodels::sph::modules
