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
 * @file PhantomDumpEOSUtils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/io/PhantomDump.hpp"

// cf phantom
// This module contains stuff to do with the equation of state
//  Current options:
//     1 = isothermal eos
//     2 = adiabatic/polytropic eos
//     3 = eos for a locally isothermal disc as in Lodato & Pringle (2007)
//     4 = GR isothermal
//     6 = eos for a locally isothermal disc as in Lodato & Pringle (2007),
//         centered on a sink particle
//     7 = z-dependent locally isothermal eos
//     8 = Barotropic eos
//     9 = Piecewise polytrope
//    10 = MESA EoS
//    11 = isothermal eos with zero pressure
//    12 = ideal gas with radiation pressure
//    13 = locally isothermal prescription from Farris et al. (2014) generalised for generic
//         hierarchical systems
//    14 = locally isothermal prescription from Farris et al. (2014) for
//         binarysystem
//    15 = Helmholtz free energy eos 16 = Shen eos 20 = Ideal gas + radiation + various
//         forms of recombination energy from HORMONE (Hirai et al., 2020)
//

namespace shammodels::sph::phdump {

    /// check if alphau is set in the header, which is the case for (maxvxyzu >= 4)
    bool is_maxvxyzu_at_least_4(const PhantomDump &dump);

    /**
     * @brief Load the EOS1 from the phantom dump
     *
     * @param[in] dump Phantom dump file
     * @param[out] cs Sound speed
     */
    void eos1_load(const PhantomDump &dump, f64 &cs);

    /**
     * @brief Write the EOS1 to the phantom dump
     *
     * @param[out] dump Phantom dump file
     * @param[in] cs Sound speed
     */
    void eos1_write(PhantomDump &dump, const f64 &cs);

    /**
     * @brief Load the EOS2 from the phantom dump
     *
     * @param[in] dump Phantom dump file
     * @param[out] gamma Adiabatic index
     */
    void eos2_load(const PhantomDump &dump, f64 &gamma);

    /**
     * @brief Write the EOS2 to the phantom dump
     *
     * @param[out] dump Phantom dump file
     * @param[in] gamma Adiabatic index
     */
    void eos2_write(PhantomDump &dump, const f64 &gamma);

    /**
     * @brief Load the EOS3 from the phantom dump
     *
     * @param[in] dump Phantom dump file
     * @param[out] cs0 Sound speed at the reference radius
     * @param[out] q Power law index
     * @param[out] r0 Reference radius
     */
    void eos3_load(const PhantomDump &dump, f64 &cs0, f64 &q, f64 &r0);

    /**
     * @brief Write the EOS3 to the phantom dump
     *
     * @param[out] dump Phantom dump file
     * @param[in] cs0 Sound speed at the reference radius
     * @param[in] q Power law index
     * @param[in] r0 Reference radius
     */
    void eos3_write(PhantomDump &dump, const f64 &cs0, const f64 &q, const f64 &r0);

} // namespace shammodels::sph::phdump
