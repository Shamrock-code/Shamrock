// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <shamunits/constant_table.hpp>
#include <shamunits/prefix_table.hpp>
#include <shamunits/unit_table.hpp>
#include <iostream>

// g++ -std=c++20 -Iinclude exemple.cpp

int main(void) {

    using namespace shamunits;

    // create si units
    constexpr UnitSystem<double> si{};

    // get the value of au^2 in the unit system
    // but it is quite big :)
    std::cout << si.get<upow(units::astronomical_unit, 2)>() << std::endl;

    /*
     * create a unit system with time in Myr, length in au, mass in solar masses
     */
    constexpr UnitSystem<double> astro{
        si.get<prefix::mega * units::year>(),
        si.get<units::astronomical_unit>(),
        si.get<constants::sol_mass>(),
    };

    // this time it returns 1 because the base length is the astronomical unit
    std::cout << astro.get<upow(units::astronomical_unit, 2)>() << std::endl;

    // in those units G is 3.94781e+13
    std::cout << astro.get<constants::G>() << std::endl;

    // units compose: one km/s is 210950 au/Myr
    std::cout << astro.get<prefix::kilo * units::metre / units::second>() << std::endl;

    // a value returned by the code in astro units (here 12 Myr) converted to seconds:
    // prints 3.78691e+14 s
    double t = 12;
    std::cout << t * astro.to<units::second>() << " s" << std::endl;

    // units, prefixes and constants can also be looked up by name at run time:
    // one parsec is 206266 au
    std::cout << astro.get(units::from_name("pc")) << std::endl;

    // Constants<T> is kept for backward compatibility: this is astro.get<constants::G>()
    std::cout << Constants<double>{astro}.get<constants::G>() << std::endl;
}
