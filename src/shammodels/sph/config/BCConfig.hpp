// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file BCConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
 
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <variant>

namespace shammodels::sph {

    template<class Tvec>
    struct BCConfig;

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::BCConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    struct Free {
        Tscal expand_tolerance = 1.2;
    };
    struct Periodic {};
    struct ShearingPeriodic {
        i32_3 shear_base;
        i32_3 shear_dir;
        Tscal shear_speed;
    };

    using Variant = std::variant<Free, Periodic, ShearingPeriodic>;

    Variant config = Free{};

    inline void set_free() { config = Free{}; }

    inline void set_periodic() { config = Periodic{}; }

    inline void set_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        config = ShearingPeriodic{shear_base, shear_dir, speed};
    }


    inline void print_status() {
        logger::raw_ln("--- Bondaries config");

        if (Free *v = std::get_if<Free>(&config)) {
            logger::raw_ln("  Config Type : Free boundaries");
        } else if (Periodic *v = std::get_if<Periodic>(&config)) {
            logger::raw_ln("  Config Type : Periodic boundaries");
        } else if (ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&config)) {
            logger::raw_ln("  Config Type : ShearingPeriodic (Stone 2010)");
            logger::raw_ln("  shear_base   =", v->shear_base);
            logger::raw_ln("  shear_dir   =", v->shear_dir);
            logger::raw_ln("  shear_speed =", v->shear_speed);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- Bondaries config config (deduced)");

        logger::raw_ln("-------------");
    }
};