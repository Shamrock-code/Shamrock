// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PhysicsModeFactory.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Implementation of PhysicsMode factory
 *
 * Creates the appropriate PhysicsMode implementation based on the physics config.
 * Each factory function takes the specific config type and creates the corresponding mode.
 */

#include "shambase/exception.hpp"
#include "shammodels/gsph/core/PhysicsModeFactory.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianMode.hpp"
#include "shammodels/gsph/physics/sr/SRMode.hpp"

namespace shammodels::gsph::core {

    template<class Tvec, template<class> class SPHKernel>
    std::unique_ptr<PhysicsMode<Tvec, SPHKernel>> PhysicsModeFactory::create_newtonian(
        const physics::NewtonianConfig<Tvec> & /*config*/) {
        // PhysicsMode gets config through evolve_timestep/init_fields - no need at construction
        return std::make_unique<physics::newtonian::NewtonianMode<Tvec, SPHKernel>>();
    }

    template<class Tvec, template<class> class SPHKernel>
    std::unique_ptr<PhysicsMode<Tvec, SPHKernel>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<Tvec> &config) {
        return std::make_unique<physics::sr::SRMode<Tvec, SPHKernel>>(config);
    }

    template<class Tvec, template<class> class SPHKernel>
    std::unique_ptr<PhysicsMode<Tvec, SPHKernel>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<Tvec> & /*config*/) {
        shambase::throw_with_loc<std::runtime_error>(
            "MHD physics mode not yet implemented. Coming soon!");
        return nullptr;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Explicit template instantiations
    // ════════════════════════════════════════════════════════════════════════════

    using namespace shammath;

    // M-spline kernels (Monaghan)
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M4>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M4>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M4>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<sycl::vec<double, 3>> &);

    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M6>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M6>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M6>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<sycl::vec<double, 3>> &);

    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M8>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M8>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, M8>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<sycl::vec<double, 3>> &);

    // Wendland kernels (C2, C4, C6)
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C2>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C2>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C2>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<sycl::vec<double, 3>> &);

    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C4>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C4>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C4>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<sycl::vec<double, 3>> &);

    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C6>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C6>> PhysicsModeFactory::create_sr(
        const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, C6>> PhysicsModeFactory::create_mhd(
        const physics::MHDConfig<sycl::vec<double, 3>> &);

    // Truncated Gaussian kernel (TGauss3) - preferred for SR-GSPH
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, TGauss3>> PhysicsModeFactory::
        create_newtonian(const physics::NewtonianConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, TGauss3>> PhysicsModeFactory::
        create_sr(const physics::SRConfig<sycl::vec<double, 3>> &);
    template std::unique_ptr<PhysicsMode<sycl::vec<double, 3>, TGauss3>> PhysicsModeFactory::
        create_mhd(const physics::MHDConfig<sycl::vec<double, 3>> &);

} // namespace shammodels::gsph::core
