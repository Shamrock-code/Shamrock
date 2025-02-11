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
 * @file qabConfig.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::sph{
    using Tscal = shambase::VecComponent<Tvec>;
    auto lambda_qav =
                            [](Tscal rho, Tscal cs, Tscal v_scal_rhat, Tscal alpha_AV, Tscal beta_AV) {
                                Tscal abs_v_ab_r_ab = sycl::fabs(v_scal_rhat);
                                Tscal vsig          = alpha_AV * cs + beta_AV * abs_v_ab_r_ab;
                                return sham::max(-Tscal(0.5) * rho * vsig * v_scal_rhat, Tscal(0));
                            };

}