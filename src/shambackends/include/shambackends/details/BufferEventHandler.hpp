// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file BufferEventHandler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambackends/sycl.hpp"

namespace sham::details {

    class BufferEventHandler {


        public:

        /**
         * @brief Vector of events related to read operations on the buffer
         *
         * This vector keeps track of the events that correcpond to read operation on the managed object
         */
        std::vector<sycl::event> read_events;

        /**
         * @brief Vector of events related to write operations on the buffer
         *
         * This vector keeps track of the events that correcpond to wire operation on the managed object
         */
        std::vector<sycl::event> write_events;

        
        /**
         * @brief Wait for all the buffer accesses to be completed
         *
         * This function waits for all the buffer accesses to be completed. It
         * waits for both read and write events to be completed.
         *
         * @param src_loc Source location of the call to this function
         */
        void wait_all(SourceLocation src_loc = SourceLocation{}) {
            sycl::event::wait(read_events);
            sycl::event::wait(write_events);

            read_events.clear();
            write_events.clear();
        }

       
        bool is_empty() { return read_events.empty() && write_events.empty(); }

        enum last_op { READ, WRITE } last_access;

        bool up_to_date_events = true;

        void read_access(std::vector<sycl::event> &depends_list, SourceLocation src_loc = SourceLocation{});

        void write_access(std::vector<sycl::event> &depends_list, SourceLocation src_loc = SourceLocation{});

        void complete_state(sycl::event e, SourceLocation src_loc = SourceLocation{});
    };
    
} // namespace sham::details
