// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file stacktrace.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include <sstream>
#include <vector>
#include <string>


namespace shambase::details {

    struct ChromeProfileEntry {
        std::string name;
        u64 time_val;
        bool is_start;

        std::string format(u32 world_rank);
    };


    std::string ChromeProfileEntry::format(u32 world_rank) {
        if (is_start) {
            return shambase::format_printf(R"({
                "cat": "%s",  
                "pid": %d,  
                "tid": %d, 
                "ts": %zu, 
                "ph": "B", 
                "name": "%s", 
                "args": { 
                }
            })",
                name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());

        } else {
            return shambase::format_printf(R"({
                "cat": "%s",  
                "pid": %d,  
                "tid": %d, 
                "ts": %zu, 
                "ph": "E", 
                "name": "%s", 
                "args": { 
                }
            })",
                name.c_str(),
                world_rank,
                world_rank,
                time_val,
                name.c_str());
        }
    }


    std::vector<ChromeProfileEntry> profile_data_chrome;

    inline void add_entry_chrome(std::source_location loc, f64 time, bool is_start){
        auto to_prof_time = [](f64 in){
            return static_cast<u64>(in * 1e6);
        };
        profile_data_chrome.push_back(ChromeProfileEntry{
            loc.function_name(), to_prof_time(time), is_start
        });
    }

    inline void clear_chrome_entry(){
        profile_data_chrome.clear();
    }

    void dump_profilings_chrome(std::string process_prefix, u32 world_rank){

        std::ofstream outfile(process_prefix + std::to_string(world_rank));
        outfile << "[";

        u32 len = profile_data_chrome.size();

        for (u32 i = 0; i < len; i++) {
            outfile << profile_data_chrome[i].format(world_rank);
            if (i != len - 1) {
                outfile << ",";
            }
        }

        outfile << "]";
        outfile.close();

    }
}


namespace shambase::details {

    auto make_timer = []() -> Timer {
        Timer tmp;
        tmp.start();
        return tmp;
    };

    Timer global_timer = make_timer();

    //two entry types, 
    // one with start, end 
    // one with start, end as separate envents


    struct ProfileEntry{
        f64 time_start;
        f64 time_end;
        std::string entry_name;

        std::string format(){
            return shambase::format_printf(R"({"tstart": %f, "tend": %f, "name": "%s"})", time_start,time_end,entry_name);
        }
    };

    std::vector<ProfileEntry> profile_data;

    f64 get_wtime(){
        global_timer.end();
        return global_timer.elasped_sec(); 
    }

    void register_profile_entry_start(std::source_location loc, f64 start_time){
        add_entry_chrome(loc, start_time, true);
    };

    void register_profile_entry(std::source_location loc, f64 start_time, f64 end_time){
        profile_data.push_back({start_time, end_time, loc.function_name()});
        add_entry_chrome(loc, end_time, false);
    };

    void clear_profiling_data(){
        profile_data.clear();
        clear_chrome_entry();
    }

    void dump_profilings(std::string process_prefix, u32 world_rank){
        std::ofstream outfile(process_prefix + std::to_string(world_rank));
        outfile << "[";

        u32 len = profile_data.size();

        for (u32 i = 0; i < len; i++) {
            outfile << profile_data[i].format();
            if (i != len - 1) {
                outfile << ",";
            }
        }

        outfile << "]";
        outfile.close();
    }

}

namespace shambase {

    /**
     * @brief get the formatted callstack
     *
     * @return std::string
     */
    std::string fmt_callstack() {
        std::stack<SourceLocation> cpy = details::call_stack;

        std::vector<std::string> lines;

        while (!cpy.empty()) {
            SourceLocation l = cpy.top();
            lines.push_back(l.format_one_line_func());
            cpy.pop();
        }

        std::reverse(lines.begin(), lines.end());

        std::stringstream ss;
        for (u32 i = 0; i < lines.size(); i++) {
            ss << shambase::format(" {:2} : {}\n", i, lines[i]);
        }

        return ss.str();
    }

} // namespace shambase