// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <iostream>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>

// g++ -std=c++17 -Iinclude exemple.cpp

int main(void){

    using namespace shamunits;

    //create si units
    UnitSystem<double> si {};

    // get the value of au^2 in the unit system
    // but it is quite big :)
    std::cout << si.get<units::astronomical_unit,2>() << std::endl;

    double sol_mass = Constants<double>(si).sol_mass();

    /*
    * create a unit system with time in Myr, lenght in au, mass in solar masses
    */
    UnitSystem<double> astro_units {
        si.get<mega, units::years>(),
        si.get<units::astronomical_unit>(),
        si.get<units::kilogramm>()*sol_mass,
    };

    //this time it returns 1 because the base lenght is the astronomical unit
    std::cout << astro_units.get<units::astronomical_unit,2>() << std::endl;

    Constants<double> astro_cte {astro_units};

    // in those units G is 3.94781e+25
    std::cout << astro_cte.G() << std::endl;

}