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
 * @file NodeMacro.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)

 * @brief A macro to simplify node edge declarations and definitions.
 * It allows to declare read-only and read-write edges in a concise way,
 * and to set and get them easily.
 *
 */

#define DECL_RO(type, name) const type &name;
#define DECL_RW(type, name) type & name;
#define PARAM_RO(type, name) std::shared_ptr<type> name,
#define PARAM_RW(type, name) std::shared_ptr<type> name,
#define PUSH_RO1(type, name) name,
#define PUSH_RW1(type, name)
#define PUSH_RO2(type, name)
#define PUSH_RW2(type, name) name,
#define GET_RO(type, name) get_ro_edge<type>(ro++),
#define GET_RW(type, name) get_rw_edge<type>(rw++),

#define EXPAND_NODE_EDGES(EDGES)                                                                   \
                                                                                                   \
    struct Edges {                                                                                 \
        EDGES(DECL_RO, DECL_RW)                                                                    \
    };                                                                                             \
                                                                                                   \
    inline void set_edges(EDGES(PARAM_RO, PARAM_RW) SourceLocation loc = SourceLocation{}) {       \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        __internal_set_ro_edges({EDGES(PUSH_RO1, PUSH_RW1)});                                      \
        __internal_set_rw_edges({EDGES(PUSH_RO2, PUSH_RW2)});                                      \
    }                                                                                              \
                                                                                                   \
    inline Edges get_edges() {                                                                     \
        int ro = 0;                                                                                \
        int rw = 0;                                                                                \
        return Edges{EDGES(GET_RO, GET_RW)};                                                       \
    }
