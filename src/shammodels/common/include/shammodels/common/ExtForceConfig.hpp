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
 * @file ExtForceConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/math.hpp"
#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <type_traits>
#include <string>
#include <variant>

namespace shammodels {

    template<class Tvec>
    struct ExtForceVariant {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        struct PointMass {
            Tscal central_mass;
            Tscal Racc;
        };

        struct LenseThirring {
            Tscal central_mass;
            Tscal Racc;
            Tscal a_spin;
            Tvec dir_spin;
        };

        /**
         * \brief Shearing box forces as in athena
         * \cite Stone2010_shear_box
         * \f[
         *  \mathbf{f} = 2\Omega_0 \left(  q \Omega_0 x +  v_y \right) \basevec{x} -2\Omega_0 v_x
         * \basevec{y} - \Omega_0^2 z \basevec{z}  \f] Shear speed : \f[ \omega = q \Omega_0 L_x \f]
         */
        struct ShearingBoxForce {
            i32_3 shear_base = {1, 0, 0};
            i32_3 shear_dir  = {0, 1, 0};

            Tscal Omega_0;
            Tscal eta;
            Tscal q;

            inline Tscal shear_speed(Tscal box_length) { return q * Omega_0 * box_length; }

            ShearingBoxForce() = default;
            ShearingBoxForce(Tscal Omega_0, Tscal eta, Tscal q)
                : Omega_0(Omega_0), eta(eta), q(q) {};
            ShearingBoxForce(i32_3 shear_base, i32_3 shear_dir, Tscal Omega_0, Tscal eta, Tscal q)
                : shear_base(shear_base), shear_dir(shear_dir), Omega_0(Omega_0), eta(eta), q(q) {};
        };

        using VariantForce = std::variant<PointMass, LenseThirring, ShearingBoxForce>;
        VariantForce val;
    };

    template<class Tvec>
    struct ExtForceConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using PointMass        = typename ExtForceVariant<Tvec>::PointMass;
        using LenseThirring    = typename ExtForceVariant<Tvec>::LenseThirring;
        using ShearingBoxForce = typename ExtForceVariant<Tvec>::ShearingBoxForce;

        using VariantForce = std::variant<PointMass, LenseThirring, ShearingBoxForce>;

        std::vector<ExtForceVariant<Tvec>> ext_forces;

        inline void add_point_mass(Tscal central_mass, Tscal Racc) {
            ext_forces.push_back(ExtForceVariant<Tvec>{PointMass{central_mass, Racc}});
        }

        inline void
        add_lense_thirring(Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
            if (sham::abs(sycl::length(dir_spin) - 1) > 1e-8) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the sping direction should be a unit vector");
            }
            ext_forces.push_back(
                ExtForceVariant<Tvec>{LenseThirring{central_mass, Racc, a_spin, dir_spin}});
        }

        /**
         * @brief
         * \todo add check for norm of shear vectors
         */
        inline void add_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {

            ext_forces.push_back(ExtForceVariant<Tvec>{ShearingBoxForce{Omega_0, eta, q}});
        }
    };

} // namespace shammodels

namespace shammodels {
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ExtForceVariant<Tvec> &p) {
        using T = ExtForceVariant<Tvec>;

        using PointMass        = typename T::PointMass;
        using LenseThirring    = typename T::LenseThirring;
        using ShearingBoxForce = typename T::ShearingBoxForce;

        if (const PointMass *v = std::get_if<PointMass>(&p.val)) {
            j = {
                {"force_type", "point_mass"}, {"central_mass", v->central_mass}, {"Racc", v->Racc}};
        } else if (const LenseThirring *v = std::get_if<LenseThirring>(&p.val)) {
            j = {
                {"force_type", "lense_thirring"},
                {"central_mass", v->central_mass},
                {"Racc", v->Racc},
                {"a_spin", v->a_spin},
                {"dir_spin", v->dir_spin},
            };
        } else if (const ShearingBoxForce *v = std::get_if<ShearingBoxForce>(&p.val)) {
            j = {
                {"force_type", "shearing_box_force"},
                {"shear_base", v->shear_base},
                {"shear_dir", v->shear_dir},
                {"Omega_0", v->Omega_0},
                {"eta", v->eta},
                {"q", v->q},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ExtForceVariant<Tvec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;
        using T     = ExtForceVariant<Tvec>;

        if (!j.contains("force_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        std::string force_type;
        j.at("force_type").get_to(force_type);

        using PointMass        = typename T::PointMass;
        using LenseThirring    = typename T::LenseThirring;
        using ShearingBoxForce = typename T::ShearingBoxForce;

        if (force_type == "point_mass") {
            p.val = PointMass{
                j.at("central_mass").get<Tscal>(),
                j.at("Racc").get<Tscal>(),
            };
        } else if (force_type == "lense_thirring") {
            p.val = LenseThirring{
                j.at("central_mass").get<Tscal>(),
                j.at("Racc").get<Tscal>(),
                j.at("a_spin").get<Tscal>(),
                j.at("dir_spin").get<Tvec>(),
            };
        } else if (force_type == "shearing_box_force") {
            p.val = ShearingBoxForce{
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("Omega_0").get<Tscal>(),
                j.at("eta").get<Tscal>(),
                j.at("q").get<Tscal>(),
            };
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ExtForceConfig<Tvec> &p) {
        using T = ExtForceConfig<Tvec>;

        j = {{"force_list", p.ext_forces}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ExtForceConfig<Tvec> &p) {
        using T = ExtForceConfig<Tvec>;

        j.at("force_list").get_to(p.ext_forces);
    }
} // namespace shammodels
