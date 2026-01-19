// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file interface_handler.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/patch/interfaces/interface_handler.hpp"

template<>
void LegacyInterfacehandler<f32_3, f32>::comm_interfaces(PatchScheduler &sched, bool periodic) {
    StackEntry stack_loc{};
    impl::comm_interfaces<f32_3, f32>(sched, interface_comm_list, interface_map, periodic);
}

template<>
void LegacyInterfacehandler<f64_3, f64>::comm_interfaces(PatchScheduler &sched, bool periodic) {
    StackEntry stack_loc{};
    impl::comm_interfaces<f64_3, f64>(sched, interface_comm_list, interface_map, periodic);
}
