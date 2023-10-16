// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file interface_handler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "interface_handler.hpp"


template <> void LegacyInterfacehandler<f32_3, f32>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    StackEntry stack_loc{};
    impl::comm_interfaces<f32_3, f32>(sched, interface_comm_list, interface_map,periodic);
    
    
}

template <> void LegacyInterfacehandler<f64_3, f64>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    StackEntry stack_loc{};
    impl::comm_interfaces<f64_3, f64>(sched, interface_comm_list, interface_map,periodic);
    
    
}
