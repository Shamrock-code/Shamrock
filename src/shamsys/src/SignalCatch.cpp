// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SignalCatch.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shambase/stacktrace.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

#include <csignal>

namespace shamsys::details {
    void signal_callback_handler(int signum) {

        auto get_signame = [&]() -> std::string {
            if (signum == SIGKILL) {
                return "SIGKILL";
            }
            if (signum == SIGTERM) {
                return "SIGTERM";
            }
            if (signum == SIGINT) {
                return "SIGINT";
            }
            return std::to_string(signum);
        };

        std::cout << "... received signal world rank=" + std::to_string(shamcomm::world_rank()) +
                         " : " + get_signame() + "\ncurrent stacktrace : \n" +
                         shambase::fmt_callstack()
                  << std::endl;

        //std::cout << "dump profiling : " << std::endl;

        
        #ifdef SHAMROCK_USE_PROFILING
        //shambase::details::dump_profiling(shamcomm::world_rank());
        #endif


        std::cout << "exiting ... " << std::endl;
        // Terminate program
        exit(signum);
    }
} // namespace shamsys::details

namespace shamsys {
    void register_signals() {
        signal(SIGTERM, details::signal_callback_handler);
        signal(SIGINT, details::signal_callback_handler);
        signal(SIGKILL, details::signal_callback_handler);
    }
} // namespace shamsys
