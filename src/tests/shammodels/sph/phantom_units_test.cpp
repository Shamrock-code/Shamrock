// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/constants.hpp"
#include "shammodels/sph/io/Phantom2Shamrock.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"
#include <cmath>
#include <optional>

// truncated value of pi used by phantom (physcon.f90) in units.f90
static constexpr f64 phantom_pi = 3.1415926536;

NEW_TEST(Unittest, "shammodels/sph/phantom-units-cgs", 1) {
    using namespace shamunits;

    // Phantom's default disc units: au, solar mass, yr/(2 pi)
    UnitSystem<f64> si{};
    Constants<f64> si_ctes{si};
    f64 au_m    = si_ctes.au();
    f64 msol_kg = si_ctes.sol_mass();
    f64 utime_s = si_ctes.year() / (2 * shambase::constants::pi<f64>);

    std::optional<UnitSystem<f64>> units = UnitSystem<f64>(utime_s, au_m, msol_kg);

    shammodels::sph::PhantomDump dump{};
    shammodels::sph::write_shamrock_units_in_phantom_dump(units, dump, false);

    f64 udist  = dump.read_header_float<f64>("udist");
    f64 umass  = dump.read_header_float<f64>("umass");
    f64 utime  = dump.read_header_float<f64>("utime");
    f64 umagfd = dump.read_header_float<f64>("umagfd");

    // Phantom headers are in cgs (cm, g, s)
    REQUIRE_FLOAT_EQUAL(udist / 1.495978707e13, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(umass / 1.98847e33, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(utime / (3.15576e7 / (2 * shambase::constants::pi<f64>) ), 1., 1e-12);

    // phantom's units.f90:
    //   unit_charge = sqrt(umass*udist/(4 pi))
    //   umagfd      = umass/(utime*unit_charge)
    f64 ucharge_ref = std::sqrt(1.98847e33 * 1.495978707e13 / (4 * phantom_pi));
    f64 umagfd_ref  = 1.98847e33 / (utime * ucharge_ref);
    REQUIRE_FLOAT_EQUAL(umagfd / umagfd_ref, 1., 1e-12);

    // Read back, the unit system should be the one we started from
    UnitSystem<f64> read_units = shammodels::sph::get_shamrock_units<f64>(dump);

    REQUIRE_FLOAT_EQUAL(read_units.m_inv / units->m_inv, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(read_units.kg_inv / units->kg_inv, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(read_units.s_inv / units->s_inv, 1., 1e-12);

    // G = 1 in au / solar mass / (yr / 2 pi) is the defining property of phantom disc units.
    // It is not exact here since shamunits' year, au and solar mass are independent constants.
    f64 g_code = Constants<f64>(read_units).G();
    REQUIRE_FLOAT_EQUAL(g_code, 1., 1e-3);
}

NEW_TEST(Unittest, "shammodels/sph/phantom-units-cgs-no-units", 1) {
    // Without units, SI is used as the code units, written in cgs to the dump
    std::optional<shamunits::UnitSystem<f64>> units = std::nullopt;

    shammodels::sph::PhantomDump dump{};
    shammodels::sph::write_shamrock_units_in_phantom_dump(units, dump, false);

    REQUIRE_FLOAT_EQUAL(dump.read_header_float<f64>("udist"), 100., 1e-12);
    REQUIRE_FLOAT_EQUAL(dump.read_header_float<f64>("umass"), 1000., 1e-12);
    REQUIRE_FLOAT_EQUAL(dump.read_header_float<f64>("utime"), 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(dump.read_header_float<f64>("umagfd"), 11.209982432814067, 1e-12);

    shamunits::UnitSystem<f64> read_units = shammodels::sph::get_shamrock_units<f64>(dump);

    REQUIRE_FLOAT_EQUAL(read_units.m_inv, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(read_units.kg_inv, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(read_units.s_inv, 1., 1e-12);
}

NEW_TEST(Unittest, "shammodels/sph/phantom-units-cgs-read", 1) {
    using namespace shamunits;

    // Header as phantom writes it for au / solar mass units, with utime set such that G = 1
    f64 g_cgs = Constants<f64>::Si::G * 1e3; // cm3.g-1.s-2
    f64 udist = 1.495978707e13;
    f64 umass = 1.98847e33;
    f64 utime = std::sqrt(udist * udist * udist / (g_cgs * umass));

    shammodels::sph::PhantomDump dump{};
    dump.table_header_f64.add("udist", udist);
    dump.table_header_f64.add("umass", umass);
    dump.table_header_f64.add("utime", utime);
    // phantom's units.f90, ~8137.2 G for these units
    f64 umagfd = umass / (utime * std::sqrt(umass * udist / (4 * phantom_pi)));
    dump.table_header_f64.add("umagfd", umagfd);

    UnitSystem<f64> read_units = shammodels::sph::get_shamrock_units<f64>(dump);

    REQUIRE_FLOAT_EQUAL(read_units.m_inv / 1.495978707e11, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(read_units.kg_inv / 1.98847e30, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(read_units.s_inv / utime, 1., 1e-12);
    f64 g_code = Constants<f64>(read_units).G();
    REQUIRE_FLOAT_EQUAL(g_code, 1., 1e-10);

    // Writing these units back must reproduce phantom's header
    std::optional<UnitSystem<f64>> units = read_units;
    shammodels::sph::PhantomDump dump_out{};
    shammodels::sph::write_shamrock_units_in_phantom_dump(units, dump_out, false);

    REQUIRE_FLOAT_EQUAL(dump_out.read_header_float<f64>("udist") / udist, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(dump_out.read_header_float<f64>("umass") / umass, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(dump_out.read_header_float<f64>("utime") / utime, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(dump_out.read_header_float<f64>("umagfd") / umagfd, 1., 1e-12);
}
