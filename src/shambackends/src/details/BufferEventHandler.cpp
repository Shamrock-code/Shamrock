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
 
#include "shambase/exception.hpp"
#include <shambackends/details/BufferEventHandler.hpp>
#include <stdexcept>

namespace sham::details {

    void BufferEventHandler::read_access(std::vector<sycl::event> &depends_list, SourceLocation src_loc) {

        if (!up_to_date_events) {
            shambase::throw_with_loc<std::runtime_error>(
                "you have requested a read access on a buffer in an incomplete state"
                "read_access call location" + src_loc.format_one_line()
                );
        }

        up_to_date_events = false;
        last_access       = READ;

        for (sycl::event e : write_events) {
            depends_list.push_back(e);
        }

    }

    void BufferEventHandler::write_access(std::vector<sycl::event> &depends_list, SourceLocation src_loc) {

        if (!up_to_date_events) {
            shambase::throw_with_loc<std::runtime_error>(
                "you have requested a write access on a buffer in an incomplete state"
                "write_access call location : " + src_loc.format_one_line()
                );
        }

        up_to_date_events = false;
        last_access       = WRITE;

        for (sycl::event e : write_events) {
            depends_list.push_back(e);
        }
        for (sycl::event e : read_events) {
            depends_list.push_back(e);
        }

    }

    void BufferEventHandler::complete_state(sycl::event e, SourceLocation src_loc) {
        if (up_to_date_events) {
            shambase::throw_with_loc<std::runtime_error>(
                "the event state of that buffer is already complete"
                "complete_state call location : " + src_loc.format_one_line()
                );
        }

        if (last_access == READ) {

            read_events.push_back(e);
            up_to_date_events = true;

        } else if (last_access == WRITE) {

            // the new event depends on those so we can clear
            write_events.clear();
            read_events.clear();

            write_events.push_back(e);
            up_to_date_events = true;
        }
    }

} // namespace sham::details