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
 * @file BufferEventHandler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"

namespace shamalgs {
    u32 gen_buf_hash();

    struct BufferEventHandler {
        std::vector<sycl::event> event_last_read;
        sycl::event event_last_write;

        const u32 id_hash = gen_buf_hash();
        inline u32 get_hash() { return id_hash; }
        inline std::string get_hash_log() { return shambase::format("id = {} |", id_hash); }

        bool up_to_date_events = true;

        enum LastEvent { READ, READ_WRITE } last_event_create;

        void add_read_dependancies(std::vector<sycl::event> &depends_list);

        void add_read_write_dependancies(std::vector<sycl::event> &depends_list);

        void register_read_event(sycl::event e);

        void register_read_write_event(sycl::event e);

        void synchronize();
    };

} // namespace shamalgs
