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
        std::vector<sycl::event> read_dependencies;
        std::vector<sycl::event> write_dependencies;

        void wait_all(SourceLocation src_loc = SourceLocation{}) {
            sycl::event::wait(read_dependencies);
            sycl::event::wait(write_dependencies);
            read_dependencies.clear();
            write_dependencies.clear();
        }

        bool is_empty() { return read_dependencies.empty() && write_dependencies.empty(); }

        enum last_op { READ, WRITE } last_access;

        bool up_to_date_events = true;

        void read_access(std::vector<sycl::event> &depends_list, SourceLocation src_loc = SourceLocation{});

        void write_access(std::vector<sycl::event> &depends_list, SourceLocation src_loc = SourceLocation{});

        void complete_state(sycl::event e, SourceLocation src_loc = SourceLocation{});
    };
} // namespace sham::details
