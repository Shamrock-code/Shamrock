// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file EventList.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/EventList.hpp"
#include "shamcomm/logs.hpp"

sham::EventList::~EventList() {
    if (!consumed) {
        shamcomm::logs::warn_ln(
            "Backends",
            shambase::format(
                "EventList destroyed without being consumed :\n    -> creation : {}",
                loc_build.format_one_line()));
        for (auto &e : events) {
            e.wait();
        }
    }
}
