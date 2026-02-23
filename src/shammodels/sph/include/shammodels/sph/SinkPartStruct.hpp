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
 * @file SinkPartStruct.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "nlohmann/json.hpp"
#include "shambackends/vec.hpp"
namespace shammodels::sph {

    template<class Tvec>
    struct SinkParticle {

        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        Tvec pos;
        Tvec velocity;
        Tvec sph_acceleration;
        Tvec ext_acceleration;
        Tscal mass;
        Tvec angular_momentum;
        Tscal accretion_radius;
    };

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SinkParticle<Tvec> &p) {
        // Serialize EOSConfig to a json object

        using json = nlohmann::json;

        j = json{
            {"pos", p.pos},
            {"velocity", p.velocity},
            {"sph_acceleration", p.sph_acceleration},
            {"ext_acceleration", p.ext_acceleration},
            {"mass", p.mass},
            {"angular_momentum", p.angular_momentum},
            {"accretion_radius", p.accretion_radius},
        };
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SinkParticle<Tvec> &p) {

        using json = nlohmann::json;

        j.at("pos").get_to(p.pos);
        j.at("velocity").get_to(p.velocity);
        j.at("sph_acceleration").get_to(p.sph_acceleration);
        j.at("ext_acceleration").get_to(p.ext_acceleration);
        j.at("mass").get_to(p.mass);
        j.at("angular_momentum").get_to(p.angular_momentum);
        j.at("accretion_radius").get_to(p.accretion_radius);
    }

} // namespace shammodels::sph
