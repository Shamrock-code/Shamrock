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
 * @file interact_crit_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

namespace interact_crit::utils {

    template<class InteractCd, class... Args>
    inline bool
    interact_cd_cell_patch_domain(const InteractCd &cd, const bool &in_domain, Args... args) {
        bool int_crit;
        if (in_domain) {
            int_crit = InteractCd::interact_cd_cell_patch_outdomain(cd, args...);
        } else {
            int_crit = InteractCd::interact_cd_cell_patch(cd, args...);
        }
        return int_crit;
    }

} // namespace interact_crit::utils
