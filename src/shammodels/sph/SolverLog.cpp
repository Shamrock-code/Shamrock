// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverLog.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "SolverLog.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambase/exception.hpp"
#include <stdexcept>

f64 shammodels::sph::SolverLog::get_last_rate(){

    if(step_logs.size() == 0){
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    u64 last_id = step_logs.size()-1;

    f64 rate_loc = step_logs[last_id].rate;

    f64 rate_tot = shamalgs::collective::allreduce_sum(rate_loc);

    return rate_tot;

}

u64 shammodels::sph::SolverLog::get_last_obj_count(){

    if(step_logs.size() == 0){
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    u64 last_id = step_logs.size()-1;

    u64 cnt_loc = step_logs[last_id].rank_count;

    u64 cnt_tot = shamalgs::collective::allreduce_sum(cnt_loc);

    return cnt_tot;

}