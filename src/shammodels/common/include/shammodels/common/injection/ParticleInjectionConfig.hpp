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
 * @file ParticleInjectionConfig.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Configuration of the particle injectors, shared by the SPH and GSPH solvers.
 *
 */

#include "shambase/exception.hpp"
#include "shambase/numeric_limits.hpp"
#include "sham/format/format.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace shammodels::common::injection {

    // Injected particles are tagged in their id (part_id field), so that they can be told apart
    // from the particles of the setup (ids below 2^61) without any extra field:
    //   live particle     : part_id_injected_flag | injector index << shift | particle index
    //   boundary particle : part_id_boundary_flag | injector index << shift | slot
    // Boundary particles are removed and re-created every step.

    /// Bit of the particle id flagging a particle created by an injector
    inline constexpr u64 part_id_injected_flag = u64(1) << 62;
    /// Bit of the particle id flagging a boundary particle of an injector
    inline constexpr u64 part_id_boundary_flag = u64(1) << 61;
    /// Position of the injector index in the id of an injected particle
    inline constexpr u32 part_id_injector_shift = 48;
    /// Largest particle index (per injector) that can be stored in the id
    inline constexpr u64 part_id_max_index = (u64(1) << part_id_injector_shift) - 1;
    /// Largest number of injectors whose index can be stored in the id
    inline constexpr u32 max_injector_count = u32(1) << (61 - part_id_injector_shift);

    /**
     * @brief Mass fallback rate of a tidal disruption event
     *
     * \f[
     *   \dot M(t) = \dot M_{\rm peak} \left(\frac{t - t_{\rm d}}{t_{\rm mb}}\right)^{n}
     *   \quad \text{for } t_{\rm d} + t_{\rm mb} \le t < t_{\rm end}
     * \f]
     * and zero otherwise.
     */
    template<class Tscal>
    struct FallbackRate {
        Tscal mdot_peak; // Rate at the start of the fallback, t_start = t_disrupt + t_mb
        Tscal t_mb;      // Return time of the most bound debris, counted from t_disrupt
        Tscal t_disrupt; // Time of the disruption
        Tscal index;     // Power law index of the rate
        Tscal t_end;     // Time at which the injection stops (+inf for never)

        /// Time at which the fallback starts
        inline Tscal t_start() const { return t_disrupt + t_mb; }

        /// Mass rate at time t
        inline Tscal mdot(Tscal t) const {
            if (t < t_start() || t >= t_end) {
                return 0;
            }
            return mdot_peak * std::pow((t - t_disrupt) / t_mb, index);
        }

        /// Mass injected between t_start and t
        inline Tscal injected_mass(Tscal t) const {
            Tscal t_clamped = sham::min(t, t_end); // make sure the injection stops at tend
            if (t_clamped <= t_start()) {
                return 0;
            }
            Tscal tau = (t_clamped - t_disrupt) / t_mb;
            if (index == Tscal(-1)) {
                return mdot_peak * t_mb * std::log(tau);
            }
            return mdot_peak * t_mb * (std::pow(tau, index + 1) - 1) / (index + 1);
        }

        /// Total mass injected by the end of the fallback (+inf if unbounded)
        inline Tscal total_mass() const {
            if (std::isfinite(t_end)) {
                return injected_mass(t_end);
            }
            if (index < Tscal(-1)) {
                return mdot_peak * t_mb / (-(index + 1));
            }
            return shambase::get_infty<Tscal>();
        }

        /**
         * @brief Time at which the injected mass reaches mass, inverse of injected_mass
         *
         * @return the time, or +inf if the fallback never injects that much mass
         */
        inline Tscal time_at_injected_mass(Tscal mass) const {
            if (mass <= 0) {
                return t_start();
            }
            if (mass > total_mass()) {
                return shambase::get_infty<Tscal>();
            }

            Tscal tau;
            if (index == Tscal(-1)) {
                tau = std::exp(mass / (mdot_peak * t_mb));
            } else {
                Tscal x = 1 + mass * (index + 1) / (mdot_peak * t_mb);
                if (x <= 0) {
                    return shambase::get_infty<Tscal>();
                }
                tau = std::pow(x, 1 / (index + 1));
            }

            return sham::min(t_disrupt + t_mb * tau, t_end);
        }

        inline void check() const {
            if (!(mdot_peak > 0) || !std::isfinite(mdot_peak)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    sham::format("the fallback rate needs mdot_peak > 0, got {}", mdot_peak));
            }
            if (!(t_mb > 0) || !std::isfinite(t_mb)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    sham::format("the fallback rate needs t_mb > 0, got {}", t_mb));
            }
            if (!std::isfinite(t_disrupt) || !std::isfinite(index)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the fallback rate needs a finite t_disrupt and mdot_index");
            }
            if (!(t_end > t_start())) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "the fallback rate needs t_end > t_disrupt + t_mb, got t_end = {} and "
                    "t_disrupt + t_mb = {}",
                    t_end,
                    t_start()));
            }
        }
    };

    /**
     * @brief Stream of gas injected through a disc at a fallback rate
     *
     * The stream is a conveyor belt of walls: the cross-section of the stream (a disc of radius
     * stream_radius centered on inject_pos and orthogonal to inject_vel) is sampled by a square
     * lattice, and a new wall of particles crosses the injection point each time the fallback rate
     * has supplied the mass of a wall. The lattice spacing follows the density of the stream \f$
     * \rho = \dot M / (\pi R^2 v) \f$ so that the particle distribution stays isotropic as the rate
     * decreases. The walls that have not crossed the injection point yet are kept upstream as
     * boundary particles, to give the youngest walls of the stream a correct pressure.
     */
    template<class Tvec>
    struct FallbackStream {
        using Tscal = shambase::VecComponent<Tvec>;

        Tvec inject_pos;     // Center of the injection disc
        Tvec inject_vel;     // Velocity of the injected gas
        Tscal stream_radius; // Radius of the stream cross-section
        Tscal u_inject;      // Specific internal energy of the injected gas
        FallbackRate<Tscal> rate;
        u32 n_boundary_walls; // Number of upcoming walls kept upstream as boundary particles

        inline void check() const {
            rate.check();
            if (!(stream_radius > 0) || !std::isfinite(stream_radius)) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "the fallback stream needs stream_radius > 0, got {}", stream_radius));
            }
            Tscal speed = sycl::length(inject_vel);
            if (!(speed > 0) || !std::isfinite(speed)) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "the fallback stream needs a non zero inject_vel, got {}", inject_vel));
            }
            if (!(u_inject >= 0) || !std::isfinite(u_inject)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    sham::format("the fallback stream needs u_inject >= 0, got {}", u_inject));
            }
            if (!std::isfinite(sycl::length(inject_pos))) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the fallback stream needs a finite inject_pos");
            }
        }
    };

    /// The list of the particle injectors of a solver
    template<class Tvec>
    struct ParticleInjectionConfig {
        using Tscal      = shambase::VecComponent<Tvec>;
        using injector_t = std::variant<FallbackStream<Tvec>>;

        std::vector<injector_t> injector_list;

        inline bool has_injectors() const { return !injector_list.empty(); }

        inline void add_fallback_stream(const FallbackStream<Tvec> &stream) {
            stream.check();
            injector_list.push_back(stream);
            try {
                check_config();
            } catch (...) {
                // leave the config as it was before the call
                injector_list.pop_back();
                throw;
            }
        }

        inline void check_config() const {
            if (injector_list.size() > max_injector_count) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "at most {} particle injectors are supported, got {}",
                    max_injector_count,
                    injector_list.size()));
            }
            for (const injector_t &injector : injector_list) {
                std::visit(
                    [](const auto &inj) {
                        inj.check();
                    },
                    injector);
            }
        }
    };

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ParticleInjectionConfig<Tvec> &p) {
        j = nlohmann::json::array();
        for (const auto &injector : p.injector_list) {
            if (const FallbackStream<Tvec> *s = std::get_if<FallbackStream<Tvec>>(&injector)) {
                nlohmann::json js{
                    {"type", "fallback_stream"},
                    {"inject_pos", s->inject_pos},
                    {"inject_vel", s->inject_vel},
                    {"stream_radius", s->stream_radius},
                    {"u_inject", s->u_inject},
                    {"mdot_peak", s->rate.mdot_peak},
                    {"t_mb", s->rate.t_mb},
                    {"t_disrupt", s->rate.t_disrupt},
                    {"mdot_index", s->rate.index},
                    {"n_boundary_walls", s->n_boundary_walls},
                };
                // json has no infinity, a missing t_end means the injection never stops
                if (std::isfinite(s->rate.t_end)) {
                    js["t_end"] = s->rate.t_end;
                }
                j.push_back(js);
            } else {
                shambase::throw_unimplemented();
            }
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ParticleInjectionConfig<Tvec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;

        p.injector_list.clear();
        for (const auto &item : j) {
            std::string type = item.at("type").get<std::string>();
            if (type == "fallback_stream") {
                FallbackStream<Tvec> s;
                item.at("inject_pos").get_to(s.inject_pos);
                item.at("inject_vel").get_to(s.inject_vel);
                item.at("stream_radius").get_to(s.stream_radius);
                item.at("u_inject").get_to(s.u_inject);
                item.at("mdot_peak").get_to(s.rate.mdot_peak);
                item.at("t_mb").get_to(s.rate.t_mb);
                item.at("t_disrupt").get_to(s.rate.t_disrupt);
                item.at("mdot_index").get_to(s.rate.index);
                s.rate.t_end = item.value("t_end", shambase::get_infty<Tscal>());
                item.at("n_boundary_walls").get_to(s.n_boundary_walls);
                p.injector_list.push_back(s);
            } else {
                shambase::throw_with_loc<std::runtime_error>(
                    "unknown particle injector type: " + type);
            }
        }
    }

} // namespace shammodels::common::injection
