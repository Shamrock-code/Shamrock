// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BufferEventHandler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
 
#include <shambackends/details/BufferEventHandler.hpp>

namespace sham::details {

    void BufferEventHandler::read_access(std::vector<sycl::event> &depends_list) {

        if (!up_to_date_events) {
            throw "";
        }

        up_to_date_events = false;
        last_access       = READ;

        for (sycl::event e : write_dependencies) {
            depends_list.push_back(e);
        }

    }

    void BufferEventHandler::write_access(std::vector<sycl::event> &depends_list) {

        if (!up_to_date_events) {
            throw "";
        }

        up_to_date_events = false;
        last_access       = WRITE;

        for (sycl::event e : write_dependencies) {
            depends_list.push_back(e);
        }
        for (sycl::event e : read_dependencies) {
            depends_list.push_back(e);
        }

    }

    void BufferEventHandler::complete_state(sycl::event e) {
        if (up_to_date_events) {
            throw "";
        }

        if (last_access == READ) {

            read_dependencies.push_back(e);
            up_to_date_events = true;

        } else if (last_access == WRITE) {

            // the new event depends on those so we can clear
            write_dependencies.clear();
            read_dependencies.clear();

            write_dependencies.push_back(e);
            up_to_date_events = true;
        }
    }

} // namespace sham::details