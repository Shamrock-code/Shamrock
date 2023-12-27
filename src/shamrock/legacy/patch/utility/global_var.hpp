// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file global_var.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

enum GlobalVariableType{
    min,max,sum
};

template<GlobalVariableType vartype, class T>
class GlobalVariable{

    std::unordered_map<u64, T> val_map;

    bool is_reduced = false;

    T final_val;

    public:


    template<class Lambda>
    inline void compute_var_patch(PatchScheduler & sched, Lambda && compute_fct){


        //sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer &pdat_buf) {
        sched.for_each_patch_data([&](u64 id_patch, shamrock::patch::Patch cur_p, shamrock::patch::PatchData &pdat) {
            static_assert(
                std::is_same<
                    decltype(compute_fct(id_patch,pdat)),
                    T>::value
                , "lambda funct should return the Global variable type");
            
            val_map[id_patch] = compute_fct(id_patch,pdat);

        });

    }

    void reduce_val();

    inline T get_val(){
        if(!is_reduced){
            throw shambase::make_except_with_loc<std::runtime_error>("Global value has not been reduced");
        }
        return final_val;
    }

};