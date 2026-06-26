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
 * @file LifetimeTracker.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/aliases_int.hpp"
namespace shamrock::solvergraph {

    template<typename T>
    class LifetimeTracker : public shambase::WithUUID<LifetimeTracker<T>, u64> {
        public:
        inline static void (*on_create)(u64 uuid)  = nullptr;
        inline static void (*on_destroy)(u64 uuid) = nullptr;

        inline static void (*on_state_update)(T &node)   = nullptr;
        inline static void (*on_op)(u64 uuid, u64 op_id) = nullptr;

        LifetimeTracker() : shambase::WithUUID<LifetimeTracker, u64>() {
            if (on_create != nullptr) {
                on_create(this->get_uuid());
            }
        };

        LifetimeTracker(const LifetimeTracker &)            = delete;
        LifetimeTracker &operator=(const LifetimeTracker &) = delete;

        LifetimeTracker(LifetimeTracker &&) noexcept            = default;
        LifetimeTracker &operator=(LifetimeTracker &&) noexcept = default;

        inline void notify_update(T &node) {
            if (on_state_update != nullptr) {
                on_state_update(node);
            }
        }

        inline void notify_op(u64 op_id) {
            if (on_op != nullptr) {
                on_op(this->get_uuid(), op_id);
            }
        }

        ~LifetimeTracker() {
            if (on_destroy != nullptr) {
                on_destroy(this->get_uuid());
            }
        };
    };

} // namespace shamrock::solvergraph
