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

#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
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

namespace shammodels::sph {
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;

        if (const Free *v = std::get_if<Free>(&p.config)) {
            j = {
                {"bc_type", "free"},
            };
        } else if (const Periodic *v = std::get_if<Periodic>(&p.config)) {
            j = {
                {"bc_type", "periodic"},
            };
        } else if (const ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&p.config)) {
            j = {
                {"bc_type", "shearing_periodic"},
                {"shear_base", v->shear_base},
                {"shear_dir", v->shear_dir},
                {"shear_speed", v->shear_speed},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("bc_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        std::string bc_type;
        j.at("bc_type").get_to(bc_type);

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;

        if (bc_type == "free") {
            p.set_free();
        } else if (bc_type == "periodic") {
            p.set_periodic();
        } else if (bc_type == "shearing_periodic") {
            p.set_shearing_periodic(
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("speed").get<Tscal>());
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }
} // namespace shammodels::sph
