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
 * @file GeneratorMCTorus.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Monte Carlo generator of a polytropic torus in equilibrium
 *
 */

#include "shamalgs/collective/InvariantParallelGenerator.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamphys/PolytropicTorus.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <optional>
#include <random>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class GeneratorMCTorus : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        Config &solver_config;

        struct TorusOutput {
            sycl::vec<Tscal, 3> pos;
            /// cylindrical coordinates of pos
            Tscal r_cyl, cos_phi, sin_phi;
            /// enthalpy W of the equilibrium at pos
            Tscal W;
            Tscal rho;
        };

        Tscal pmass;

        class TorusIterator;
        TorusIterator generator;
        Tscal init_h_factor;

        /// only when MHD : amplitude of A_phi
        std::optional<Tscal> A0;
        Tscal A_cut;

        /// adiabatic index, set only for an adiabatic EOS
        std::optional<Tscal> gamma;

        static TorusIterator make_generator(
            shamphys::PolytropicTorus torus,
            Tscal part_mass,
            Tscal torus_mass,
            std::mt19937_64 eng) {
            return TorusIterator(torus, part_mass, torus_mass, eng);
        }

        public:
        /**
         * @param torus the torus equilibrium (in the solver units)
         * @param part_mass mass of each particle
         * @param torus_mass total mass of the torus, the particle count is torus_mass / part_mass
         * @param plasma_beta volume averaged int P dV / int B^2/(2 mu_0) dV of the initial
         *        poloidal field, no field if empty
         * @param A_cut A_phi = A0 * max(rho / rho0 - A_cut, 0)
         * @param eng random
         * @param init_h_factor multiplier of the smoothing length inferred from the density
         */
        GeneratorMCTorus(
            ShamrockCtx &context,
            Config &solver_config,
            shamphys::PolytropicTorus torus,
            Tscal part_mass,
            Tscal torus_mass,
            std::optional<Tscal> plasma_beta,
            Tscal A_cut,
            std::mt19937_64 eng,
            Tscal init_h_factor);

        bool is_done();

        shamrock::patch::PatchDataLayer next_n(u32 nmax);

        std::string get_name() { return "GeneratorMCTorus"; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules

template<class Tvec, template<class> class SPHKernel>
class shammodels::sph::modules::GeneratorMCTorus<Tvec, SPHKernel>::TorusIterator {

    shamphys::PolytropicTorus torus;

    /// density at the pressure maximum
    Tscal rho0;

    shamalgs::collective::InvariantParallelGenerator<std::mt19937_64> generator;

    Tscal f_func(Tscal r_cyl, Tscal z) { return r_cyl * torus.rho_shape(r_cyl, z); }

    /// upper bound of f_func in the torus used for the rejection sampling
    Tscal compute_fmax();

    /// max of f_func in the torus, set in the constructor
    Tscal fmax;

    TorusOutput next(u64 seed);

    public:
    TorusIterator(
        shamphys::PolytropicTorus torus, Tscal part_mass, Tscal torus_mass, std::mt19937_64 eng)
        : TorusIterator(torus, part_mass, torus_mass, eng, torus_mass / part_mass) {}

    TorusIterator(
        shamphys::PolytropicTorus torus,
        Tscal part_mass,
        Tscal torus_mass,
        std::mt19937_64 eng,
        u64 Npart)
        : torus(torus), rho0(torus_mass / torus.volume_integral()), generator(eng, Npart) {

        shamlog_debug_ln(
            "GeneratorMCTorus",
            "part_mass",
            part_mass,
            "torus_mass",
            torus_mass,
            "Npart",
            Npart,
            "rho0",
            rho0);

        fmax = compute_fmax();

        shamlog_debug_ln("GeneratorMCTorus", "fmax", fmax);
    }

    inline const shamphys::PolytropicTorus &get_torus() const { return torus; }
    inline Tscal get_rho0() const { return rho0; }

    inline bool is_done() { return generator.is_done(); }

    inline std::vector<TorusOutput> next_n(u64 nmax) {
        std::vector<u64> seeds = generator.next_n(nmax);
        std::vector<TorusOutput> ret{};
        for (u64 seed : seeds) {
            ret.push_back(next(seed));
        }
        return ret;
    }
};
