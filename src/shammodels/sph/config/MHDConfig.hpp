@@ -0,0 +1,93 @@
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MHDConfig.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels::sph {

    template<class Tvec>
    struct MHDConfig;
}

template<class Tvec>
struct shammodels::sph::MHDConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    struct None {};

    struct IdealMHD_constrained_hyper_para{
        Tscal sigma_mhd = 0.1;
        Tscal alpha_u = 1.;
    };

    struct NonIdealMHD{
        Tscal sigma_mhd = 0.1;
        Tscal alpha_u = 1.;
    };


    // how to set a new state of a variant as a dummy:
    //a) do everything right
    //b) forget to add the state to the variant
    //-> question your life choices
    using Variant = std::variant<None, IdealMHD_constrained_hyper_para, NonIdealMHD>;

    Variant config = IdealMHD_constrained_hyper_para{};

    void set(Variant v) { config = v; }

    inline bool has_B_field() {
    bool is_B =
        bool(std::get_if<IdealMHD_constrained_hyper_para>(&config)) || bool(std::get_if<NonIdealMHD>(&config));
    return is_B;
    }

    inline bool has_psi_field() {
    bool is_psi =
        bool(std::get_if<IdealMHD_constrained_hyper_para>(&config)) || bool(std::get_if<NonIdealMHD>(&config));
    return is_psi;
    }

    inline bool has_curlB_field() {
        bool is_curlB = bool(std::get_if<NonIdealMHD>(&config));
        return is_curlB;
    }


    inline void print_status() {
        logger::raw_ln("--- MHD config");

        if (None *v = std::get_if<None>(&config)) {
            logger::raw_ln("  Config MHD Type : None (No MHD)");
        } else if (IdealMHD_constrained_hyper_para *v = std::get_if<IdealMHD_constrained_hyper_para>(&config)) {
            logger::raw_ln("  Config MHD  : Ideal MHD, constrained hyperbolic/parabolic treatment");
            logger::raw_ln("  sigma_mhd  =", v->sigma_mhd);
        } else if (NonIdealMHD *v = std::get_if<NonIdealMHD>(&config)) {
            logger::raw_ln("  Config MHD Type : Non Ideal MHD");
            logger::raw_ln("  sigma_mhd   =", v->sigma_mhd);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- MHD config (deduced)");

        logger::raw_ln("-------------");
    }
};