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
 * @file enum_TensileCorrection.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief SPMHD tensile instability correction enum + string conversion
 */

#include "shambase/exception.hpp"
#include <stdexcept>
#include <string>

namespace shammodels {

    /**
     * @brief Prefactor of the tensile instability correction -Bhat (div B) in the momentum
     * equation
     */
    enum class TensileCorrection {
        /// Bhat = B/2 regardless of the plasma beta (Borve et al. 2004)
        Borve,
        /// Bhat = f(beta) B, with f = 1 for beta < 2, (10 - beta)/8 for 2 < beta < 10, 0 otherwise
        /// (Price et al. 2018, Phantom paper, eq. 178)
        PhantomBetaSwitch,
    };

    /// Convert a TensileCorrection to its string representation (used in JSON & logs)
    inline std::string tensile_correction_to_string(TensileCorrection t) {
        switch (t) {
        case TensileCorrection::Borve            : return "borve";
        case TensileCorrection::PhantomBetaSwitch: return "phantom_beta_switch";
        }
        shambase::throw_unimplemented();
        return "";
    }

    /// Parse a TensileCorrection from its string representation (used in JSON)
    inline TensileCorrection tensile_correction_from_string(const std::string &s) {
        if (s == "borve") {
            return TensileCorrection::Borve;
        } else if (s == "phantom_beta_switch") {
            return TensileCorrection::PhantomBetaSwitch;
        }
        shambase::throw_with_loc<std::invalid_argument>(
            "unknown tensile correction '" + s
            + "', valid values are 'borve' and 'phantom_beta_switch'");
        return TensileCorrection::Borve;
    }

} // namespace shammodels
