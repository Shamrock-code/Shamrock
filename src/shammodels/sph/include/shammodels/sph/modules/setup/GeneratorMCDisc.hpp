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
 * @file GeneratorMCDisc.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/collective/InvariantParallelGenerator.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class GeneratorMCDisc : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        Config &solver_config;

        struct DiscOutput {
            sycl::vec<Tscal, 3> pos;
            sycl::vec<Tscal, 3> velocity;
            Tscal cs;
            Tscal rho;
        };

        Tscal pmass;

        class DiscIterator;
        DiscIterator generator;
        Tscal init_h_factor;

        static DiscIterator make_generator(
            Tscal part_mass,
            Tscal disc_mass,
            Tscal r_in,
            Tscal r_out,
            std::function<Tscal(Tscal)> sigma_profile,
            std::function<Tscal(Tscal)> H_profile,
            std::function<Tscal(Tscal)> rot_profile,
            std::function<Tscal(Tscal)> cs_profile,
            std::mt19937_64 eng) {
            return DiscIterator(
                part_mass,
                disc_mass,
                r_in,
                r_out,
                sigma_profile,
                H_profile,
                rot_profile,
                cs_profile,
                eng);
        }

        public:
        GeneratorMCDisc(
            ShamrockCtx &context,
            Config &solver_config,
            Tscal part_mass,
            Tscal disc_mass,
            Tscal r_in,
            Tscal r_out,
            std::function<Tscal(Tscal)> sigma_profile,
            std::function<Tscal(Tscal)> H_profile,
            std::function<Tscal(Tscal)> rot_profile,
            std::function<Tscal(Tscal)> cs_profile,
            std::mt19937_64 eng,
            Tscal init_h_factor)
            : context(context), solver_config(solver_config), generator(make_generator(
                                                                  part_mass,
                                                                  disc_mass,
                                                                  r_in,
                                                                  r_out,
                                                                  sigma_profile,
                                                                  H_profile,
                                                                  rot_profile,
                                                                  cs_profile,
                                                                  eng)),
              init_h_factor(init_h_factor), pmass(part_mass) {}

        bool is_done();

        shamrock::patch::PatchDataLayer next_n(u32 nmax);

        std::string get_name() { return "GeneratorMCDisc"; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules

template<class Tvec, template<class> class SPHKernel>
class shammodels::sph::modules::GeneratorMCDisc<Tvec, SPHKernel>::DiscIterator {

    bool done         = false;
    u64 current_index = 0;

    Tscal part_mass;
    Tscal disc_mass;
    u64 Npart;

    Tscal r_in;
    Tscal r_out;
    std::function<Tscal(Tscal)> sigma_profile;
    std::function<Tscal(Tscal)> H_profile;
    std::function<Tscal(Tscal)> rot_profile;
    std::function<Tscal(Tscal)> cs_profile;

    shamalgs::collective::InvariantParallelGenerator<std::mt19937_64> generator;

    static constexpr Tscal _2pi = 2 * shambase::constants::pi<Tscal>;

    Tscal f_func(Tscal r) { return r * sigma_profile(r); }

    DiscOutput next(u64 seed);

    public:
    DiscIterator(
        Tscal part_mass,
        Tscal disc_mass,
        Tscal r_in,
        Tscal r_out,
        std::function<Tscal(Tscal)> sigma_profile,
        std::function<Tscal(Tscal)> H_profile,
        std::function<Tscal(Tscal)> rot_profile,
        std::function<Tscal(Tscal)> cs_profile,
        std::mt19937_64 eng)
        : DiscIterator(
              part_mass,
              disc_mass,
              r_in,
              r_out,
              sigma_profile,
              H_profile,
              rot_profile,
              cs_profile,
              eng,
              disc_mass / part_mass) {}

    DiscIterator(
        Tscal part_mass,
        Tscal disc_mass,
        Tscal r_in,
        Tscal r_out,
        std::function<Tscal(Tscal)> sigma_profile,
        std::function<Tscal(Tscal)> H_profile,
        std::function<Tscal(Tscal)> rot_profile,
        std::function<Tscal(Tscal)> cs_profile,
        std::mt19937_64 eng,
        u64 Npart)
        : part_mass(part_mass), disc_mass(disc_mass), Npart(Npart), r_in(r_in), r_out(r_out),
          sigma_profile(sigma_profile), H_profile(H_profile), rot_profile(rot_profile),
          cs_profile(cs_profile), generator(eng, Npart), current_index(0) {

        shamlog_debug_ln(
            "GeneratorMCDisc",
            "part_mass",
            part_mass,
            "disc_mass",
            disc_mass,
            "r_in",
            r_in,
            "r_out",
            r_out,
            "Npart",
            Npart);
    }

    inline bool is_done() {
        return generator.is_done();
    } // just to make sure the result is not tempered with

    inline std::vector<DiscOutput> next_n(u64 nmax) {
        std::vector<u64> seeds = generator.next_n(nmax);
        std::vector<DiscOutput> ret{};
        for (u64 seed : seeds) {
            ret.push_back(next(seed));
        }
        return ret;
    }
};
