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
 * @brief Callback API to track the lifetime, state updates and operations of solvergraph objects
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/aliases_int.hpp"
#include <string_view>
#include <utility>

namespace shamrock::solvergraph {

    /**
     * @brief Tracks the lifetime of an object of type T and notifies observers through static
     * callbacks.
     *
     * Held by the tracked object as a plain value member (not a base class), so
     * trace_state_update() can take a `T&` to the enclosing object.
     *
     * Move-safety comes from the base class: a moved-from instance's is_alive() reports false,
     * so it won't emit a duplicate destroy notification.
     *
     * All callbacks are `nullptr` by default, so tracking-disabled cost is one null check per
     * notification site.
     *
     * @tparam T The tracked object type (e.g. INode, IEdge)
     */
    template<typename T>
    class LifetimeTracker : public shambase::WithUUID<LifetimeTracker<T>, u64> {
        public:
        /// Called when a tracked object is created
        inline static void (*on_create)(u64 uuid) = nullptr;
        /// Called when a tracked object is destroyed
        inline static void (*on_destroy)(u64 uuid) = nullptr;

        /// Called when the state of a tracked object changes (e.g. edges are rebound)
        inline static void (*on_state_update)(T &object) = nullptr;
        /// Called when an operation is performed on a tracked object (e.g. evaluation)
        inline static void (*on_event)(u64 uuid, std::string_view s) = nullptr;

        /// Constructor, notifies the creation of the tracked object
        LifetimeTracker() : shambase::WithUUID<LifetimeTracker, u64>() {
            if (on_create != nullptr) {
                on_create(this->get_uuid());
            }
        };

        LifetimeTracker(const LifetimeTracker &)            = delete; ///< would duplicate the UUID
        LifetimeTracker &operator=(const LifetimeTracker &) = delete; ///< would duplicate the UUID

        /// Move constructor: transfers the uuid to `this` and invalidates `other`.
        LifetimeTracker(LifetimeTracker &&) noexcept = default;

        /// Move assignment: fires `this`'s own destroy notification (if still alive), then
        /// transfers `other`'s uuid over and invalidates `other`.
        inline LifetimeTracker &operator=(LifetimeTracker &&other) noexcept {
            if (this != &other) {
                trace_destroy();
                shambase::WithUUID<LifetimeTracker, u64>::operator=(std::move(other));
            }
            return *this;
        }

        /// Notifies creation of the tracked object.
        inline void trace_create() {
            if (on_create) {
                on_create(this->uuid);
            }
        }

        /// Fires the destroy notification, if not already fired or moved from. Idempotent.
        inline void trace_destroy() {
            if (this->is_alive()) {
                if (on_destroy) {
                    on_destroy(this->uuid);
                }
                this->invalidate();
            }
        }

        // notify and update of the owning object.
        inline void trace_state_update(T &object) {
            if (this->is_alive() && on_state_update) {
                on_state_update(object);
            }
        }

        /// Use it like tracker.trace_event([]() {return "evaluate_begin";});
        /// This patern allow for almost no overhead if tracing is disabled
        template<class F>
        inline void trace_event(F &&event_info_builder) {
            if (this->is_alive() && on_event) {
                on_event(this->uuid, event_info_builder());
            }
        }

        /// Destructor, notifies destruction (unless already notified, or moved from).
        ~LifetimeTracker() { trace_destroy(); };
    };

} // namespace shamrock::solvergraph
