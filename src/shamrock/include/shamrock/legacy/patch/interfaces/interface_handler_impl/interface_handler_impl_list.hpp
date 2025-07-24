// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file interface_handler_impl_list.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/patch/simulation_domain.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"

//%Impl status : Clean unfinished

enum InterfacehandlerImpl { Tree_Send };

// forward definition of the interface handler

template<InterfacehandlerImpl impl_type, class pos_prec, class Tree>
class Interfacehandler {

    public:
    using flt = pos_prec;
    using vec = sycl::vec<flt, 3>;

    // template<class Func_interactcrit>
    // void compute_interface_list(PatchScheduler &sched, SerialPatchTree<vec> & sptree,
    // SimulationDomain<flt> & bc,std::unordered_map<u64, RadixTree> & rtrees, Func_interactcrit &&
    // interact_crit, Args & ... args);

    // TODO
    void initial_fetch();

    void fetch_field();

    template<class Function>
    void for_each_interface(u64 patch_id, Function &&fct);
};
