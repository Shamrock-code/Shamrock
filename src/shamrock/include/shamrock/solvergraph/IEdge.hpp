// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file IEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shamrock/solvergraph/IFreeable.hpp"
#include "shamrock/solvergraph/LifetimeTracker.hpp"
#include <string>

namespace shamrock::solvergraph {

    class INode;

    class IEdge : public IFreeable {

        /// track the UUID of the node to notify changes.
        /// Sadly we can not use a WithUUID directly since it is hard to log destruction in the
        /// destructor if the object was moved which could lead to double free notifications.
        std::shared_ptr<LifetimeTracker<IEdge>> tracker
            = std::make_shared<LifetimeTracker<IEdge>>();

        public:
        inline std::string get_label() const { return _impl_get_dot_label(); }
        inline std::string get_tex_symbol() const { return _impl_get_tex_symbol(); }

        virtual std::string _impl_get_dot_label() const  = 0;
        virtual std::string _impl_get_tex_symbol() const = 0;

        /// Get the UUID of the node
        inline u64 get_uuid() const { return shambase::get_check_ref(tracker).get_uuid(); }

        inline virtual ~IEdge() {}
    };

} // namespace shamrock::solvergraph
