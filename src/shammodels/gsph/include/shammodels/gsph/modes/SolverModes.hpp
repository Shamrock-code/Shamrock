// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverModes.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solver mode modifiers for GSPH
 *
 * Defines how the solver evolves. Modes can modify the computed
 * derivatives before time integration.
 */

namespace shammodels::gsph::modes {

    /**
     * @brief Normal time evolution mode
     *
     * No modifications to computed derivatives. This is the default mode.
     */
    struct NormalEvolution {
        static constexpr bool has_convergence_check = false;

        template<class Derivs, class State>
        static void modify_derivatives(Derivs & /* derivs */, const State & /* state */) {}
    };

} // namespace shammodels::gsph::modes
