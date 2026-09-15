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
 * @file ExtForceConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
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

        struct PN_PW {
            Tscal central_mass;
            Tvec central_pos;
            Tscal Racc;
        };

        struct PN_1PN {
            Tscal central_mass;
            Tvec central_pos;
            Tvec central_vel;
        };

        struct LenseThirring {
            Tscal central_mass;
            Tscal Racc;
            Tscal a_spin;
            Tvec dir_spin;
            Tvec central_pos;
            Tvec central_vel;
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

        /// f = -GMy / sqrt(R0^2 + y^2)
        struct VerticalDiscPotential {
            Tscal central_mass;
            Tscal R0;
        };

        /// f = -eta v
        struct VelocityDissipation {
            Tscal eta;
        };

        using VariantForce = std::variant<
            PointMass,
            PN_PW,
            PN_1PN,
            LenseThirring,
            ShearingBoxForce,
            VerticalDiscPotential,
            VelocityDissipation>;
        VariantForce val;
    };

    template<class Tvec>
    struct ExtForceConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using PointMass             = typename ExtForceVariant<Tvec>::PointMass;
        using PN_PW                 = typename ExtForceVariant<Tvec>::PN_PW;
        using PN_1PN                = typename ExtForceVariant<Tvec>::PN_1PN;
        using LenseThirring         = typename ExtForceVariant<Tvec>::LenseThirring;
        using ShearingBoxForce      = typename ExtForceVariant<Tvec>::ShearingBoxForce;
        using VerticalDiscPotential = typename ExtForceVariant<Tvec>::VerticalDiscPotential;
        using VelocityDissipation   = typename ExtForceVariant<Tvec>::VelocityDissipation;

        std::vector<ExtForceVariant<Tvec>> ext_forces;

        inline void add_point_mass(Tscal central_mass, Tscal Racc) {
            ext_forces.push_back(ExtForceVariant<Tvec>{PointMass{central_mass, Racc}});
        }

        inline void add_paczynski_wiita(Tscal central_mass, Tvec central_pos, Tscal Racc) {
            ext_forces.push_back(ExtForceVariant<Tvec>{PN_PW{central_mass, central_pos, Racc}});
        }

        inline void add_1pn(Tscal central_mass, Tvec central_pos, Tvec central_vel) {
            ext_forces.push_back(
                ExtForceVariant<Tvec>{PN_1PN{central_mass, central_pos, central_vel}});
        }

        inline void add_lense_thirring(
            Tscal central_mass,
            Tscal Racc,
            Tscal a_spin,
            Tvec dir_spin,
            Tvec central_pos = Tvec{},
            Tvec central_vel = Tvec{}) {
            if (sham::abs(sycl::length(dir_spin) - 1) > 1e-8) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the sping direction should be a unit vector");
            }
            ext_forces.push_back(ExtForceVariant<Tvec>{
                LenseThirring{central_mass, Racc, a_spin, dir_spin, central_pos, central_vel}});
        }

        /**
         * @brief
         * \todo add check for norm of shear vectors
         */
        inline void add_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {

            ext_forces.push_back(ExtForceVariant<Tvec>{ShearingBoxForce{Omega_0, eta, q}});
        }

        inline void add_vertical_disc_potential(Tscal central_mass, Tscal R0) {
            ext_forces.push_back(ExtForceVariant<Tvec>{VerticalDiscPotential{central_mass, R0}});
        }

        inline void add_velocity_dissipation(Tscal eta) {
            ext_forces.push_back(ExtForceVariant<Tvec>{VelocityDissipation{eta}});
        }
    };

} // namespace shammodels

namespace shammodels {
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ExtForceVariant<Tvec> &p) {
        using T = ExtForceVariant<Tvec>;

        using PointMass             = typename T::PointMass;
        using PN_PW                 = typename T::PN_PW;
        using PN_1PN                = typename T::PN_1PN;
        using LenseThirring         = typename T::LenseThirring;
        using ShearingBoxForce      = typename T::ShearingBoxForce;
        using VerticalDiscPotential = typename T::VerticalDiscPotential;
        using VelocityDissipation   = typename T::VelocityDissipation;

        if (const PointMass *v = std::get_if<PointMass>(&p.val)) {
            j = {
                {"force_type", "point_mass"}, {"central_mass", v->central_mass}, {"Racc", v->Racc}};

        } else if (const PN_PW *v = std::get_if<PN_PW>(&p.val)) {
            j
                = {{"force_type", "paczynski_wiita"},
                   {"central_mass", v->central_mass},
                   {"central_pos", v->central_pos},
                   {"Racc", v->Racc}};
        } else if (const PN_1PN *v = std::get_if<PN_1PN>(&p.val)) {
            j = {{"force_type", "1pn"},
                 {"central_mass", v->central_mass},
                 {"central_pos", v->central_pos},
                 {"central_vel", v->central_vel}};
        } else if (const LenseThirring *v = std::get_if<LenseThirring>(&p.val)) {
            j = {
                {"force_type", "lense_thirring"},
                {"central_mass", v->central_mass},
                {"Racc", v->Racc},
                {"a_spin", v->a_spin},
                {"dir_spin", v->dir_spin},
                {"central_pos", v->central_pos},
                {"central_vel", v->central_vel},
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
        } else if (const VerticalDiscPotential *v = std::get_if<VerticalDiscPotential>(&p.val)) {
            j
                = {{"force_type", "vertical_disc_potential"},
                   {"central_mass", v->central_mass},
                   {"R0", v->R0}};
        } else if (const VelocityDissipation *v = std::get_if<VelocityDissipation>(&p.val)) {
            j = {{"force_type", "velocity_dissipation"}, {"eta", v->eta}};
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

        using PointMass             = typename T::PointMass;
        using PN_PW                 = typename T::PN_PW;
        using PN_1PN                = typename T::PN_1PN;
        using LenseThirring         = typename T::LenseThirring;
        using ShearingBoxForce      = typename T::ShearingBoxForce;
        using VerticalDiscPotential = typename T::VerticalDiscPotential;
        using VelocityDissipation   = typename T::VelocityDissipation;

        if (force_type == "point_mass") {
            p.val = PointMass{
                j.at("central_mass").get<Tscal>(),
                j.at("Racc").get<Tscal>(),
            };
        } else if (force_type == "paczynski_wiita") {
            p.val = PN_PW{
                j.at("central_mass").get<Tscal>(),
                j.at("central_pos").get<Tvec>(),
                j.at("Racc").get<Tscal>(),
            };
        } else if (force_type == "1pn") {
            p.val = PN_1PN{
                j.at("central_mass").get<Tscal>(),
                j.value("central_pos", Tvec{}),
                j.value("central_vel", Tvec{}),
            };
        } else if (force_type == "lense_thirring") {
            p.val = LenseThirring{
                j.at("central_mass").get<Tscal>(),
                j.at("Racc").get<Tscal>(),
                j.at("a_spin").get<Tscal>(),
                j.at("dir_spin").get<Tvec>(),
                j.value("central_pos", Tvec{}),
                j.value("central_vel", Tvec{}),
            };
        } else if (force_type == "shearing_box_force") {
            p.val = ShearingBoxForce{
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("Omega_0").get<Tscal>(),
                j.at("eta").get<Tscal>(),
                j.at("q").get<Tscal>(),
            };
        } else if (force_type == "vertical_disc_potential") {
            p.val = VerticalDiscPotential{
                j.at("central_mass").get<Tscal>(),
                j.at("R0").get<Tscal>(),
            };
        } else if (force_type == "velocity_dissipation") {
            p.val = VelocityDissipation{j.at("eta").get<Tscal>()};
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
