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
 * @file IFieldRefsAny.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Type erased view on an IFieldRefs<T>, dispatching back to the concrete field type.
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include <type_traits>
#include <utility>

namespace shamrock::solvergraph {

    template<class T>
    class IFieldRefs;

    // clang-format off
    /// True if T is one of the field types enabled in the scheduler
    template<class T>
    inline constexpr bool is_enabled_field_type_v =
        #define X(args) std::is_same_v<T, args> ||
        XMAC_LIST_ENABLED_FIELD false
        #undef X
        ;
    // clang-format on

    /**
     * @brief Visitor over the field types enabled in the scheduler.
     *
     * Prefer `visit_field_refs` over implementing this by hand, there is one overload per enabled
     * type (see XMAC_LIST_ENABLED_FIELD).
     */
    class IFieldRefsVisitor {
        public:
        virtual ~IFieldRefsVisitor() = default;

#define X(args) virtual void visit(IFieldRefs<args> &field) = 0;
        XMAC_LIST_ENABLED_FIELD
#undef X
    };

    /// Const variant of IFieldRefsVisitor
    class IFieldRefsConstVisitor {
        public:
        virtual ~IFieldRefsConstVisitor() = default;

#define X(args) virtual void visit(const IFieldRefs<args> &field) = 0;
        XMAC_LIST_ENABLED_FIELD
#undef X
    };

    /**
     * @brief Type erased handle on an `IFieldRefs<T>`, dispatching back to the concrete T.
     *
     * A container of fields (see `EdgeList`) is homogeneous in its element type, but a list of
     * fields to dump, checkpoint or diagnose generally mixes types. Mixing this interface into
     * `IFieldRefs<T>` gives a common element type for such a list, and the concrete type is
     * recovered by double dispatch, without the list itself knowing what it will be used for.
     *
     * @code{.cpp}
     * fields.for_each([&](u32 i, const IFieldRefsAny &any) {
     *     visit_field_refs(any, [&](const auto &refs) {
     *         // refs is a const IFieldRefs<T> & for the concrete T
     *     });
     * });
     * @endcode
     *
     * @note This is deliberately not an IEdge, it is a mixin added on top of the existing edge
     * hierarchy so that no diamond appears (`IFieldRefs<T>` already reaches IEdge through
     * `IFieldSpan<T>`).
     */
    class IFieldRefsAny {
        public:
        virtual ~IFieldRefsAny() = default;

        /// Dispatch to the visitor overload matching the concrete field type
        virtual void accept(IFieldRefsVisitor &visitor) = 0;

        /// Const variant of accept
        virtual void accept(IFieldRefsConstVisitor &visitor) const = 0;
    };

    /// Adapts a generic functor `f(IFieldRefs<T> &)` into an IFieldRefsVisitor
    template<class Func>
    class FieldRefsLambdaVisitor : public IFieldRefsVisitor {
        Func f;

        public:
        explicit FieldRefsLambdaVisitor(Func &&f) : f(std::forward<Func>(f)) {}

#define X(args)                                                                                    \
    inline void visit(IFieldRefs<args> &field) override { f(field); }
        XMAC_LIST_ENABLED_FIELD
#undef X
    };

    /// Const variant of FieldRefsLambdaVisitor
    template<class Func>
    class FieldRefsLambdaConstVisitor : public IFieldRefsConstVisitor {
        Func f;

        public:
        explicit FieldRefsLambdaConstVisitor(Func &&f) : f(std::forward<Func>(f)) {}

#define X(args)                                                                                    \
    inline void visit(const IFieldRefs<args> &field) override { f(field); }
        XMAC_LIST_ENABLED_FIELD
#undef X
    };

    /**
     * @brief Recover the concrete field type of a type erased field.
     *
     * @param any the type erased field
     * @param f a generic functor, it is instantiated for every enabled field type
     */
    template<class Func>
    inline void visit_field_refs(IFieldRefsAny &any, Func &&f) {
        FieldRefsLambdaVisitor<Func> visitor{std::forward<Func>(f)};
        any.accept(visitor);
    }

    /// Const variant of visit_field_refs
    template<class Func>
    inline void visit_field_refs(const IFieldRefsAny &any, Func &&f) {
        FieldRefsLambdaConstVisitor<Func> visitor{std::forward<Func>(f)};
        any.accept(visitor);
    }

} // namespace shamrock::solvergraph
