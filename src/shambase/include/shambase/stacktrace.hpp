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
#include "shambase/aliases_float.hpp"
#include "shambase/string.hpp"
#include <stack>

#ifdef SHAMROCK_USE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

namespace shambase::details {
    
    f64 get_wtime();

    #ifdef SHAMROCK_USE_PROFILING

    void register_profile_entry_start(std::source_location loc, f64 start_time);
    void register_profile_entry(std::source_location loc, f64 start_time, f64 end_time);

    void dump_profilings(std::string process_prefix, u32 world_rank);
    void dump_profilings_chrome(std::string process_prefix, u32 world_rank);
    void clear_profiling_data();
    //void add_prof_entry(std::string n, bool is_start);
    //void dump_profiling(u32 world_rank);
    #endif

    inline std::stack<SourceLocation> call_stack;

    struct BasicStackEntry {
        SourceLocation loc;
        bool do_timer;

        #ifdef SHAMROCK_USE_PROFILING
        f64 wtime_start;
        #endif

        inline BasicStackEntry(bool do_timer = true, SourceLocation &&loc = SourceLocation{})
            : loc(loc), do_timer(do_timer) {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer){
                wtime_start = get_wtime();
                register_profile_entry_start(loc.loc, wtime_start);
            }
            #endif
            call_stack.emplace(loc);
            #ifdef SHAMROCK_USE_NVTX
            nvtxRangePush(loc.loc.function_name());
            #endif
        }

        inline ~BasicStackEntry() {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer){
                f64 wtime_end = get_wtime();
                register_profile_entry(loc.loc, wtime_start, wtime_end);
            }
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

        #ifdef SHAMROCK_USE_PROFILING
        f64 wtime_start;
        #endif

        inline NamedBasicStackEntry(std::string name,
                                    bool do_timer        = true,
                                    SourceLocation &&loc = SourceLocation{})
            : name(name), loc(loc), do_timer(do_timer) {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer){
                wtime_start = get_wtime();
            }
            #endif
            call_stack.emplace(loc);
            #ifdef SHAMROCK_USE_NVTX
            nvtxRangePush(name.c_str());
            #endif
        }

        inline ~NamedBasicStackEntry() {
            #ifdef SHAMROCK_USE_PROFILING
            if (do_timer){
                f64 wtime_end = get_wtime();
                register_profile_entry(loc.loc, wtime_start, wtime_end);
            }
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