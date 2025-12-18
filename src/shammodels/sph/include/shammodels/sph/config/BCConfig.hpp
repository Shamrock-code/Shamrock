// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file BCConfig.hpp
 * @author Guo (guo.yansong@optimind.tech)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::sph {

    /**
     * @brief Boundary conditions configuration
     *
     * This struct is used to configure the boundary conditions of a simulation.
     *
     * @tparam Tvec The vector type used for the simulation.
     */
    template<class Tvec>
    struct BCConfig;

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::BCConfig {

    /// Type of the components of the vector of coordinates
    using Tscal = shambase::VecComponent<Tvec>;
    /// Number of dimensions of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief Free boundary condition
     *
     * The box will be expanded if a particle is outside of it.
     */
    struct Free {
        /**
         * @brief The tolerance for the box expansion
         *
         * If a particle is outside of the box, the box will be expanded to the new range with an
         * added margin factor of expand_tolerance
         */
        Tscal expand_tolerance = 1.2;
    };

    /**
     * @brief Periodic boundary condition
     */
    struct Periodic {};

    /**
     * @brief Shearing periodic boundary condition
     * @todo use a bib entry instead
     * @see https://ui.adsabs.harvard.edu/abs/2010ApJS..189..142S/abstract
     */

    struct ShearingPeriodic {
        /**
         * @brief The base of the scalar product to define the number of shearing periodicity to be
         * applied
         */
        i32_3 shear_base;

        /**
         * @brief The direction of the shear
         */
        i32_3 shear_dir;

        /**
         * @brief The speed of the shear
         */
        Tscal shear_speed;
    };

    /**
     * @brief Wall boundary condition with mirror particles
     *
     * Creates "wall particles" (not to be confused with "ghost particles" used for MPI)
     * beyond the domain boundaries to provide proper neighbor support for particles
     * near walls. These particles mirror the state of boundary particles with
     * the same velocity (inflow-style boundary).
     *
     * Reference: Similar to ghost particle approach in sphcode (sr_sod.cpp)
     */
    struct Wall {
        /**
         * @brief Number of wall particle layers beyond each boundary
         * More layers = better neighbor support but more particles
         * Typically 2-6 layers (2*Rkern smoothing lengths worth)
         */
        u32 num_layers = 4;

        /**
         * @brief Which boundaries have walls (bit flags)
         * Bit 0: -x wall, Bit 1: +x wall
         * Bit 2: -y wall, Bit 3: +y wall
         * Bit 4: -z wall, Bit 5: +z wall
         * Default: all walls enabled (0x3F = 0b111111)
         */
        u32 wall_flags = 0x3F;
    };

    /// Variant of all types of artificial viscosity possible
    using Variant = std::variant<Free, Periodic, ShearingPeriodic, Wall>;

    /// The actual configuration (default to free boundaries)
    Variant config = Free{};

    /// Set the boundary condition to free boundaries
    inline void set_free() { config = Free{}; }

    /// Set the boundary condition to periodic boundaries
    inline void set_periodic() { config = Periodic{}; }

    /**
     * @brief Set the boundary condition to shearing periodic boundaries
     *
     * @param shear_base The base of the scalar product to define the number of shearing periodicity
     * to be applied
     * @param shear_dir The direction of the shear
     * @param speed The speed of the shear
     */
    inline void set_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        config = ShearingPeriodic{shear_base, shear_dir, speed};
    }

    /**
     * @brief Set the boundary condition to wall boundaries
     *
     * @param num_layers Number of wall particle layers (default 4)
     * @param wall_flags Which walls to enable (default all = 0x3F)
     */
    inline void set_wall(u32 num_layers = 4, u32 wall_flags = 0x3F) {
        config = Wall{num_layers, wall_flags};
    }

    /**
     * @brief Prints the current boundary condition configuration to the logger.
     *
     * The function logs the type of boundary condition (free, periodic, shearing periodic, or wall)
     * and the relevant parameters for each case.
     */
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
        } else if (Wall *v = std::get_if<Wall>(&config)) {
            logger::raw_ln("  Config Type : Wall boundaries (mirror particles)");
            logger::raw_ln("  num_layers  =", v->num_layers);
            logger::raw_ln("  wall_flags  =", v->wall_flags, "(hex)");
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- Bondaries config config (deduced)");

        logger::raw_ln("-------------");
    }
};

namespace shammodels::sph {

    /**
     * @brief Serialize a BCConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The BCConfig to serialize
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;
        using Wall             = typename T::Wall;

        // Write the config type into the JSON object
        if (const Free *v = std::get_if<Free>(&p.config)) {
            j = {
                {"bc_type", "free"},
            };
        } else if (const Periodic *v = std::get_if<Periodic>(&p.config)) {
            j = {
                {"bc_type", "periodic"},
            };
        } else if (const ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&p.config)) {
            // Write the shear base, direction, and speed into the JSON object
            j = {
                {"bc_type", "shearing_periodic"},
                {"shear_base", v->shear_base},
                {"shear_dir", v->shear_dir},
                {"shear_speed", v->shear_speed},
            };
        } else if (const Wall *v = std::get_if<Wall>(&p.config)) {
            j = {
                {"bc_type", "wall"},
                {"num_layers", v->num_layers},
                {"wall_flags", v->wall_flags},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    /**
     * @brief Deserialize a JSON object into a BCConfig
     *
     * @param[in] j  The JSON object to read from
     * @param[out] p The BCConfig to deserialize
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Tscal = shambase::VecComponent<Tvec>;

        // Check if the JSON object contains the "bc_type" field
        if (!j.contains("bc_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field bc_type is found in this json");
        }

        // Read the config type from the JSON object
        std::string bc_type;
        j.at("bc_type").get_to(bc_type);

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;
        using Wall             = typename T::Wall;

        // Set the BCConfig based on the config type
        if (bc_type == "free") {
            p.set_free();
        } else if (bc_type == "periodic") {
            p.set_periodic();
        } else if (bc_type == "shearing_periodic") {
            p.set_shearing_periodic(
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("speed").get<Tscal>());
        } else if (bc_type == "wall") {
            u32 num_layers = j.at("num_layers").get<u32>();
            u32 wall_flags = j.at("wall_flags").get<u32>();
            p.set_wall(num_layers, wall_flags);
        } else {
            shambase::throw_unimplemented("Unknown boundary type: " + bc_type);
        }
    }

} // namespace shammodels::sph
