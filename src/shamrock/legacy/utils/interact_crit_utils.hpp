// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file interact_crit_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
