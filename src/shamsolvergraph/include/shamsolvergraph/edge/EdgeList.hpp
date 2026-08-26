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
 * @file EdgeList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the EdgeList class, an edge holding a runtime sized list of other edges.
 *
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamsolvergraph/edge/IEdgeNamed.hpp"
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace shamrock::solvergraph {

    /**
     * @brief An edge aggregating a runtime sized list of other edges.
     *
     * A node has a fixed number of slots (see `EXPAND_NODE_EDGES`), so a node that must read a
     * variable number of edges takes a single `EdgeList` slot instead. This is the same trick used
     * by `INullOptEdge` to express optional edges without changing the node arity.
     *
     * @code{.cpp}
     * #define NODE_EDGES(X_RO, X_RW)                                                            \
     *     X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                      \
     *     X_RO(shamrock::solvergraph::EdgeList<shamrock::solvergraph::IFieldSpan<Tscal>>, fields)
     *
     * // in _impl_evaluate_internal
     * edges.fields.for_each([&](u32 i, const shamrock::solvergraph::IFieldSpan<Tscal> &field) {
     *     field.check_sizes(edges.sizes.indexes);
     * });
     * @endcode
     *
     * Constness of the list propagates to the entries: a `X_RO` slot yields a `const EdgeList &`,
     * whose accessors hand out `const T &`, while a `X_RW` slot yields mutable access. This is why
     * the underlying shared pointers are never exposed, as constness would not propagate through
     * them.
     *
     * @note `sham::DDMultiRef` is variadic at compile time, so a node holding N entries at runtime
     * can only issue one kernel call per entry, they can not be fused into a single kernel call
     * without extra machinery.
     *
     * @tparam T The type of the listed edges. It does not have to derive from IEdge, which allows
     * listing type erasing interfaces mixed into edges (see `IFieldRefsAny`).
     */
    template<class T>
    class EdgeList : public IEdgeNamed {

        /// The listed edges
        std::vector<std::shared_ptr<T>> entries = {};

        public:
        using IEdgeNamed::IEdgeNamed;

        virtual ~EdgeList() {}

        /**
         * @brief Set the list of edges.
         *
         * @param new_entries The edges to list, none of them can be null
         * @throws std::invalid_argument if any entry is null
         */
        inline void set_entries(std::vector<std::shared_ptr<T>> new_entries) {
            for (std::size_t i = 0; i < new_entries.size(); i++) {
                if (!bool(new_entries[i])) {
                    shambase::throw_with_loc<std::invalid_argument>(sham::format(
                        "entry {} of the edge list '{}' is a nullptr, please pass a shared pointer "
                        "with a valid edge",
                        i,
                        get_label()));
                }
            }
            entries = std::move(new_entries);
        }

        /// Get the number of listed edges
        inline u32 size() const { return static_cast<u32>(entries.size()); }

        /// Is the list empty
        inline bool empty() const { return entries.empty(); }

        /// Get the listed edge at the given index
        inline T &get(u32 i) { return shambase::get_check_ref(entries.at(i)); }

        /// Const variant of get
        inline const T &get(u32 i) const { return shambase::get_check_ref(entries.at(i)); }

        /// Apply a function to each listed edge, the signature must be `f(u32 i, T &entry)`
        template<class Func>
        inline void for_each(Func &&f) {
            for (u32 i = 0; i < size(); i++) {
                f(i, get(i));
            }
        }

        /// Const variant of for_each, the signature must be `f(u32 i, const T &entry)`
        template<class Func>
        inline void for_each(Func &&f) const {
            for (u32 i = 0; i < size(); i++) {
                f(i, get(i));
            }
        }

        /**
         * @brief Drop the list.
         *
         * Only the list is cleared, the listed edges are left untouched as they are owned by
         * whoever registered them. This matches `FieldRefs::free_alloc` or
         * `PatchDataLayerRefs::free_alloc`, which likewise only drop their references.
         */
        inline virtual void free_alloc() { entries = {}; }

        /// Expose the listed edges to the graph tooling (dot graph, ...)
        inline virtual std::vector<std::shared_ptr<IEdge>> get_sub_edges() const {
            std::vector<std::shared_ptr<IEdge>> ret{};
            ret.reserve(entries.size());
            for (const auto &entry : entries) {
                // sidecast, T may not derive from IEdge (see IFieldRefsAny)
                if (auto as_edge = std::dynamic_pointer_cast<IEdge>(entry)) {
                    ret.push_back(std::move(as_edge));
                }
            }
            return ret;
        }

        /// Make a shared pointer to an EdgeList
        static std::shared_ptr<EdgeList<T>> make_shared(std::string name, std::string texsymbol) {
            return std::make_shared<EdgeList<T>>(name, texsymbol);
        }
    };

} // namespace shamrock::solvergraph
