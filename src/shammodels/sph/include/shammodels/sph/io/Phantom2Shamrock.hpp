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
 * @file Phantom2Shamrock.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/common/EOSConfig.hpp"
#include "shammodels/sph/config/AVConfig.hpp"
#include "shammodels/sph/config/BCConfig.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamunits/UnitSystem.hpp"
#include <cstdlib>
#include <optional>

namespace shammodels::sph {

    /**
     * @brief Generate a Shamrock EOS configuration from a PhantomDump object.
     *
     * @param phdump Reference to the PhantomDump object.
     * @param bypass_error Flag to bypass error handling.
     *
     * @return The EOS configuration corresponding to the given PhantomDump object.
     *
     * @throws std::runtime_error If an error occurs during configuration retrieval.
     */
    template<class Tvec>
    EOSConfig<Tvec> get_shamrock_eosconfig(PhantomDump &phdump, bool bypass_error);

    /// Write the eos config to th phantom dump header
    template<class Tvec>
    void
    write_shamrock_eos_in_phantom_dump(EOSConfig<Tvec> &cfg, PhantomDump &dump, bool bypass_error);

    /**
     * @brief Generate an Shamrock artificial viscosity configuration from a PhantomDump object.
     *
     * @param phdump Reference to the PhantomDump object.
     *
     * @return The artificial viscosity configuration corresponding to the given PhantomDump object.
     */
    template<class Tvec>
    AVConfig<Tvec> get_shamrock_avconfig(PhantomDump &phdump);

    /**
     * @brief Get the shamrock units object
     * \todo load also magfd
     * @tparam Tscal
     * @param phdump
     * @return shamunits::UnitSystem<Tscal>
     */
    template<class Tscal>
    shamunits::UnitSystem<Tscal> get_shamrock_units(PhantomDump &phdump);

    /// Write shamrock units config into the phantom dump
    template<class Tscal>
    void write_shamrock_units_in_phantom_dump(
        std::optional<shamunits::UnitSystem<Tscal>> &units, PhantomDump &dump, bool bypass_error);

    /**
     * @brief Generate an Shamrock boundary configuration from a PhantomDump object.
     *
     * @param phdump Reference to the PhantomDump object.
     *
     * @return The boundary configuration corresponding to the given PhantomDump object.
     */
    template<class Tvec>
    BCConfig<Tvec> get_shamrock_boundary_config(PhantomDump &phdump);

    template<class Tvec>
    void write_shamrock_boundaries_in_phantom_dump(
        BCConfig<Tvec> &cfg, std::tuple<Tvec, Tvec> box_size, PhantomDump &dump, bool bypass_error);

} // namespace shammodels::sph
