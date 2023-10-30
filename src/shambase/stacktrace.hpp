// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file stacktrace.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "SourceLocation.hpp"
#include "shambase/string.hpp"
#include <stack>

#ifdef SHAMROCK_USE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

namespace shambase::details {

    #ifdef SHAMROCK_USE_PROFILING
    void add_prof_entry(std::string n, bool is_start);
    void dump_profiling(u32 world_rank);
    #endif

    inline std::stack<SourceLocation> call_stack;

    struct BasicStackEntry {
        SourceLocation loc;
        bool do_timer;

        inline BasicStackEntry(bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : loc(loc), do_timer(do_timer) {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer)
                add_prof_entry(loc.functionName, true);
            #endif
            call_stack.emplace(loc);
            #ifdef SHAMROCK_USE_NVTX
            nvtxRangePush(loc.functionName);
            #endif
        }

        inline ~BasicStackEntry() {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer)
                add_prof_entry(call_stack.top().functionName, false);
            #endif
            call_stack.pop();
            #ifdef SHAMROCK_USE_NVTX
            nvtxRangePop();
            #endif
        }
    };

    struct NamedBasicStackEntry {
        SourceLocation loc;
        bool do_timer;
        std::string name;

        inline NamedBasicStackEntry(std::string name,
                                    bool do_timer        = true,
                                    SourceLocation &&loc = SourceLocation{})
            : name(name), loc(loc), do_timer(do_timer) {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer)
                add_prof_entry(name, true);
            #endif
            call_stack.emplace(loc);
            #ifdef SHAMROCK_USE_NVTX
            nvtxRangePush(name.c_str());
            #endif
        }

        inline ~NamedBasicStackEntry() {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer)
                add_prof_entry(name, false);
            #endif
            call_stack.pop();
            #ifdef SHAMROCK_USE_NVTX
            nvtxRangePop();
            #endif
        }
    };

} // namespace shambase::details

namespace shambase {

    /**
     * @brief get the formatted callstack
     *
     * @return std::string
     */
    std::string fmt_callstack();

} // namespace shambase

using StackEntry      = shambase::details::BasicStackEntry;
using NamedStackEntry = shambase::details::NamedBasicStackEntry;