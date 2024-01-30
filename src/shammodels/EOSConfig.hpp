// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EOSConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/legacy/log.hpp"
#include <string>
#include <type_traits>
#include <variant>

namespace shammodels {

    template<class Tvec>
    struct EOSConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        // EOS types definition usable in the code

        struct Adiabatic {
            Tscal gamma = 5. / 3.;
        };

        struct LocallyIsothermal {};

        struct LocallyIsothermalLP07 {
            Tscal cs0 = 0.005;
            Tscal q = -2;
            Tscal r0 = 10;
        };

        // internal wiring of the eos to the code

        using Variant = std::variant<Adiabatic, LocallyIsothermal, LocallyIsothermalLP07>;

        Variant config = Adiabatic{};

        inline void set_adiabatic(Tscal gamma) { config = Adiabatic{gamma}; }
        inline void set_locally_isothermal() { config = LocallyIsothermal{}; }
        inline void set_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) { config = LocallyIsothermalLP07{cs0, q, r0}; }

        inline void print_status();
    };

} // namespace shammodels

template<class Tvec>
void shammodels::EOSConfig<Tvec>::print_status() {

    std::string s;
    if constexpr (std::is_same_v<f32_3, Tvec>) {
        s = "f32_3";
    }

    if constexpr (std::is_same_v<f64_3, Tvec>) {
        s = "f64_3";
    }

    logger::raw_ln("EOS config", s, ":");
    if (Adiabatic *eos_config = std::get_if<Adiabatic>(&config)) {
        logger::raw_ln("adiabatic : ");
        logger::raw_ln("gamma", eos_config->gamma);
    } else if (LocallyIsothermal *eos_config = std::get_if<LocallyIsothermal>(&config)) {
        logger::raw_ln("locally isothermal : ");
    } else if (LocallyIsothermalLP07 *eos_config = std::get_if<LocallyIsothermalLP07>(&config)) {
        logger::raw_ln("locally isothermal (Lodato Price 2007) : ");
    } else {
        shambase::throw_unimplemented();
    }
}
