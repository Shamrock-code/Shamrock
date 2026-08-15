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
 * @file sink_edges_helper.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Helpers to access SPH sink particles stored as SoA synchronized data edges.
 *
 */

#include "shambase/type_name_info.hpp"
#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SinkPartStruct.hpp"
#include "shamsolvergraph/SolverGraphSerializable.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include <vector>

namespace shambase {
    template<class T>
    struct TypeNameInfo<std::vector<T>> {
        inline static const std::string name = "std::vector<" + get_type_name<T>() + ">";
    };
} // namespace shambase

namespace shammodels::sph {

    template<class Tvec>
    struct SinkEdges {
        using Tscal = shambase::VecComponent<Tvec>;

        std::vector<Tvec> &pos;
        std::vector<Tvec> &vel;
        std::vector<Tvec> &acc_sph;
        std::vector<Tvec> &acc_ext;
        std::vector<Tscal> &mass;
        std::vector<Tvec> &angular_momentum;
        std::vector<Tscal> &accretion_radius;

        inline bool has_sinks() const { return !pos.empty(); }
        inline size_t size() const { return pos.size(); }
    };

    namespace details {

        template<class T>
        inline void ensure_sync_data_edge(
            shamrock::solvergraph::SolverGraphSerializable &sync_data,
            const std::string &name,
            const std::string &tex_symbol,
            T init_value) {

            if (!sync_data.has_edge(name)) {
                auto edge = sync_data.register_edge(
                    name, shamrock::solvergraph::IDataEdgeSerializable<T>(name, tex_symbol));
                edge->data = std::move(init_value);
            }
        }

        template<class T>
        inline T &get_sync_data(
            shamrock::solvergraph::SolverGraphSerializable &sync, const std::string &name) {
            return sync.template get_edge_ref<shamrock::solvergraph::IDataEdgeSerializable<T>>(name)
                .data;
        }

    } // namespace details

    /**
     * @brief Register sink SoA synchronized edges if missing (idempotent).
     */
    template<class Tvec>
    inline void ensure_sink_edges(shamrock::solvergraph::SolverGraphSerializable &sync) {
        using Tscal = shambase::VecComponent<Tvec>;

        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_pos", "\\bf{r}_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_vel", "\\bf{v}_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_acc_sph", "\\bf{a}_{\\mathrm{sink, sph}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_acc_ext", "\\bf{a}_{\\mathrm{sink, ext}}", {});
        details::ensure_sync_data_edge<std::vector<Tscal>>(
            sync, "sink_mass", "m_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_angular_momentum", "\\bf{L}_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tscal>>(
            sync, "sink_accretion_radius", "r_{\\mathrm{accretion}}", {});
    }

    /**
     * @brief Named SoA getters (edges must already exist; call ensure_sink_edges first).
     * Prefer these when a function only needs a subset of sink fields.
     */
    template<class Tvec>
    inline std::vector<Tvec> &get_sink_pos(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_pos");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_vel(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_vel");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_acc_sph(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_acc_sph");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_acc_ext(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_acc_ext");
    }

    template<class Tvec>
    inline std::vector<shambase::VecComponent<Tvec>> &get_sink_mass(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<shambase::VecComponent<Tvec>>>(sync, "sink_mass");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_angular_momentum(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_angular_momentum");
    }

    template<class Tvec>
    inline std::vector<shambase::VecComponent<Tvec>> &get_sink_accretion_radius(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<shambase::VecComponent<Tvec>>>(
            sync, "sink_accretion_radius");
    }

    /**
     * @brief Check whether any sinks are present by inspecting sink_pos only.
     *
     * Prefer this for early-exit checks instead of fetching the full SinkEdges struct.
     * Edges must already exist (call ensure_sink_edges first).
     */
    template<class Tvec>
    inline bool has_sinks(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return !get_sink_pos<Tvec>(sync).empty();
    }

    /**
     * @brief Fetch mutable references to the sink SoA synchronized edges.
     *
     * Edges must already exist (call ensure_sink_edges first).
     * Prefer named getters when only a subset of fields is needed.
     */
    template<class Tvec>
    inline SinkEdges<Tvec> get_sink_edges(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return SinkEdges<Tvec>{
            get_sink_pos<Tvec>(sync),
            get_sink_vel<Tvec>(sync),
            get_sink_acc_sph<Tvec>(sync),
            get_sink_acc_ext<Tvec>(sync),
            get_sink_mass<Tvec>(sync),
            get_sink_angular_momentum<Tvec>(sync),
            get_sink_accretion_radius<Tvec>(sync),
        };
    }

    /**
     * @brief Build an AoS sink list from the current SoA edges (Python API / dump helpers).
     */
    template<class Tvec>
    inline std::vector<SinkParticle<Tvec>> to_sink_particles(const SinkEdges<Tvec> &e) {
        std::vector<SinkParticle<Tvec>> out;
        out.reserve(e.size());
        for (size_t i = 0; i < e.size(); i++) {
            out.push_back(
                SinkParticle<Tvec>{
                    e.pos[i],
                    e.vel[i],
                    e.acc_sph[i],
                    e.acc_ext[i],
                    e.mass[i],
                    e.angular_momentum[i],
                    e.accretion_radius[i],
                });
        }
        return out;
    }

    /**
     * @brief Replace SoA sink edge contents from an AoS sink list (legacy dump migration).
     */
    template<class Tvec>
    inline void set_sink_particles(
        SinkEdges<Tvec> &e, const std::vector<SinkParticle<Tvec>> &sinks) {
        e.pos.clear();
        e.vel.clear();
        e.acc_sph.clear();
        e.acc_ext.clear();
        e.mass.clear();
        e.angular_momentum.clear();
        e.accretion_radius.clear();

        e.pos.reserve(sinks.size());
        e.vel.reserve(sinks.size());
        e.acc_sph.reserve(sinks.size());
        e.acc_ext.reserve(sinks.size());
        e.mass.reserve(sinks.size());
        e.angular_momentum.reserve(sinks.size());
        e.accretion_radius.reserve(sinks.size());

        for (const auto &s : sinks) {
            e.pos.push_back(s.pos);
            e.vel.push_back(s.velocity);
            e.acc_sph.push_back(s.sph_acceleration);
            e.acc_ext.push_back(s.ext_acceleration);
            e.mass.push_back(s.mass);
            e.angular_momentum.push_back(s.angular_momentum);
            e.accretion_radius.push_back(s.accretion_radius);
        }
    }

    /**
     * @brief Append one sink to the SoA edges.
     */
    template<class Tvec>
    inline void add_sink(
        SinkEdges<Tvec> &e,
        typename SinkEdges<Tvec>::Tscal mass,
        Tvec pos,
        Tvec velocity,
        typename SinkEdges<Tvec>::Tscal accretion_radius) {

        e.pos.push_back(pos);
        e.vel.push_back(velocity);
        e.acc_sph.push_back({});
        e.acc_ext.push_back({});
        e.mass.push_back(mass);
        e.angular_momentum.push_back({});
        e.accretion_radius.push_back(accretion_radius);
    }

} // namespace shammodels::sph
