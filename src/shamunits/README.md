# Shamrock units library

This is the units library in use in the Shamrock code, this repository will be updated when change are made to this library in the Shamrock monorepo.

Almost everything is marked `constexpr` in the library, so most of the conversion if possible will be opmitized away by the compiler, allowing for zero cost abstraction here :)

Here is an exemple of the usage of the units library :
```c++
#include <iostream>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>

int main(void){

    using namespace shamunits;

    //create si units
    UnitSystem<double> si {};

    // get the value of au^2 in the unit system
    // but it is quite big :)
    std::cout << si.get<units::astronomical_unit,2>() << std::endl;

    double sol_mass = Constants<double>(si).sol_mass();

    /*
    * create a unit system with time in Myr, length in au, mass in solar masses
    */
    UnitSystem<double> astro_units {
        si.get<mega, units::years>(),
        si.get<units::astronomical_unit>(),
        si.get<units::kilogram>()*sol_mass,
    };

    //this time it returns 1 because the base length is the astronomical unit
    std::cout << astro_units.get<units::astronomical_unit,2>() << std::endl;

    Constants<double> astro_cte {astro_units};

    // in those units G is 3.94781e+25
    std::cout << astro_cte.G() << std::endl;

    //now if the code return a value in astro_units
    //we can convert it to any units like so
    double value = 12; //here 12 Myr

    // print : value = 3.15576e+19 s
    std::cout << "value = "<< astro_units.to<units::second>() << " s"<< std::endl;

}

```

If you want to try here is a godbolt link : https://godbolt.org/z/5zjGMea57
