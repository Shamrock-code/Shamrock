// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PhysicsModeFactory.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Factory for creating physics mode instances
 *
 * Each physics mode owns its config. Factory creates the mode with its config.
 * Config structs are in physics/{newtonian,sr,mhd}/ subfolders.
 */

#include "shammodels/gsph/core/PhysicsMode.hpp"
#include "shammodels/gsph/physics/mhd/MHDConfig.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianConfig.hpp"
#include "shammodels/gsph/physics/sr/SRConfig.hpp"
#include <memory>

namespace shammodels::gsph::core {

    /**
     * @brief Factory class for creating PhysicsMode instances
     *
     * Each create function takes the specific config type and creates
     * the corresponding PhysicsMode implementation that owns that config.
     */
    class PhysicsModeFactory {
        public:
        /**
         * @brief Create Newtonian physics mode
         */
        template<class Tvec, template<class> class SPHKernel>
        static std::unique_ptr<PhysicsMode<Tvec, SPHKernel>> create_newtonian(
            const physics::NewtonianConfig<Tvec> &config);

        /**
         * @brief Create Special Relativistic physics mode
         */
        template<class Tvec, template<class> class SPHKernel>
        static std::unique_ptr<PhysicsMode<Tvec, SPHKernel>> create_sr(
            const physics::SRConfig<Tvec> &config);

        /**
         * @brief Create MHD physics mode
         */
        template<class Tvec, template<class> class SPHKernel>
        static std::unique_ptr<PhysicsMode<Tvec, SPHKernel>> create_mhd(
            const physics::MHDConfig<Tvec> &config);
    };

} // namespace shammodels::gsph::core
