// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GeneratorMCDisc.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
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
            std::mt19937 eng) {
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
            std::mt19937 eng,
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

        shamrock::patch::PatchData next_n(u32 nmax);

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

    std::mt19937 eng;

    static constexpr Tscal _2pi = 2 * shambase::constants::pi<Tscal>;

    Tscal f_func(Tscal r) { return r * sigma_profile(r); }

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
        std::mt19937 eng)
        : part_mass(part_mass), disc_mass(disc_mass), Npart(disc_mass / part_mass), r_in(r_in),
          r_out(r_out), sigma_profile(sigma_profile), H_profile(H_profile),
          rot_profile(rot_profile), cs_profile(cs_profile), eng(eng), current_index(0) {

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

        if (Npart == 0) {
            done = true;
        }
    }

    inline bool is_done() { return done; } // just to make sure the result is not tempered with

    DiscOutput next();

    inline std::vector<DiscOutput> next_n(u32 nmax) {
        std::vector<DiscOutput> ret{};
        for (u32 i = 0; i < nmax; i++) {
            if (done) {
                break;
            }

            ret.push_back(next());
        }
        return ret;
    }

    inline void skip(u32 n) {
        for (u32 i = 0; i < n; i++) {
            if (done) {
                break;
            }
            next();
        }
    }
};
