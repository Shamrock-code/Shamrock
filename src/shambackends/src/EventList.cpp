// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
