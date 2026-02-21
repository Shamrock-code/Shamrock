// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverConfig.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/io/json_print_diff.hpp"
#include "shamrock/io/json_std_optional.hpp"
#include "shamrock/io/json_utils.hpp"
#include "shamrock/io/units_json.hpp"

namespace shammodels::basegodunov {

    template<class Tvec, class TgridVec>
    inline void SolverConfig<Tvec, TgridVec>::set_layout(
        shamrock::patch::PatchDataLayerLayout &pdl) {
        pdl.add_field<TgridVec>("cell_min", 1);
        pdl.add_field<TgridVec>("cell_max", 1);
        pdl.add_field<Tscal>("rho", AMRBlock::block_size);
        pdl.add_field<Tvec>("rhovel", AMRBlock::block_size);
        pdl.add_field<Tscal>("rhoetot", AMRBlock::block_size);

        if (is_dust_on()) {
            u32 ndust = dust_config.ndust;
            pdl.add_field<Tscal>("rho_dust", (ndust * AMRBlock::block_size));
            pdl.add_field<Tvec>("rhovel_dust", (ndust * AMRBlock::block_size));
        }

        if (is_gravity_on()) {
            pdl.add_field<Tscal>("phi", AMRBlock::block_size);
        }

        if (is_gas_passive_scalar_on()) {
            u32 npscal_gas = npscal_gas_config.npscal_gas;
            pdl.add_field<Tscal>("rho_gas_pscal", (npscal_gas * AMRBlock::block_size));
        }
    }

    template<class Tvec, class TgridVec>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, TgridVec> &p) {

        j = nlohmann::json{
            {"type_id", shambase::get_type_name<Tvec>()},
            {"courant_safety_factor", p.Csafe},
            {"dust_riemann_solver", p.dust_config.dust_riemann_config},
            {"eos_gamma", p.eos_gamma},
            {"face_half_time_interpolation", p.face_half_time_interpolation},
            {"gravity_solver", p.gravity_config.gravity_mode},
            {"grid_coord_to_pos_fact", p.grid_coord_to_pos_fact},
            {"hydro_riemann_solver", p.riemann_config},
            {"passive_scalar_mode", p.npscal_gas_config.npscal_gas},
            {"slope_limiter", p.slope_config},
            {"time_state", p.time_state},
            {"unit_sys", p.unit_sys}};
    }

    template<class Tvec, class TgridVec>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, TgridVec> &p) {
        using T = SolverConfig<Tvec, TgridVec>;

        if (j.contains("type_id")) {

            std::string type_id = j.at("type_id").get<std::string>();

            if (type_id != shambase::get_type_name<Tvec>()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                    + " but got " + type_id);
            }
        }

        bool has_used_defaults  = false;
        bool has_updated_config = false;

        auto _get_to_if_contains = [&](const std::string &key, auto &value) {
            shamrock::get_to_if_contains(j, key, value, has_used_defaults);
        };

        // actual data stored in the json
        _get_to_if_contains("courant_safety_factor", p.Csafe);
        _get_to_if_contains("dust_riemann_solver", p.dust_config.dust_riemann_config);
        _get_to_if_contains("eos_gamma", p.eos_gamma);
        _get_to_if_contains("face_half_time_interpolation", p.face_half_time_interpolation);
        _get_to_if_contains("gravity_solver", p.gravity_config.gravity_mode);
        _get_to_if_contains("grid_coord_to_pos_fact", p.grid_coord_to_pos_fact);
        _get_to_if_contains("hydro_riemann_solver", p.riemann_config);
        _get_to_if_contains("passive_scalar_mode", p.npscal_gas_config.npscal_gas);
        _get_to_if_contains("slope_limiter", p.slope_config);
        _get_to_if_contains("time_state", p.time_state);
        _get_to_if_contains("unit_sys", p.unit_sys);

        if (has_used_defaults || has_updated_config) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "Ramses::SolverConfig",
                    shamrock::log_json_changes(p, j, has_used_defaults, has_updated_config));
            }
        }
    }

    template void to_json<f64_3, i64_3>(nlohmann::json &j, const SolverConfig<f64_3, i64_3> &p);
    template void from_json<f64_3, i64_3>(const nlohmann::json &j, SolverConfig<f64_3, i64_3> &p);

} // namespace shammodels::basegodunov

template class shammodels::basegodunov::SolverConfig<f64_3, i64_3>;
