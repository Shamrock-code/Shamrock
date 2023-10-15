// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchDataLayout.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "PatchDataLayout.hpp"

namespace shamrock::patch{
std::string PatchDataLayout::get_description_str(){
    std::stringstream ss;

    if(fields.empty()){
        ss << "empty table\n";
    }else{

    

    u32 index= 0;
    for(var_t & v : fields){
        v.visit([&](auto & field){
            using f_t = typename std::remove_reference<decltype(field)>::type;
            using base_t = typename f_t::field_T;

            ss << index << " : " << field.name << " : nvar=" << field.nvar << " type : ";

            if (std::is_same<base_t, f32   >::value){ss << "f32   ";}
            else if (std::is_same<base_t, f32_2 >::value){ss << "f32_2 ";}
            else if (std::is_same<base_t, f32_3 >::value){ss << "f32_3 ";}
            else if (std::is_same<base_t, f32_4 >::value){ss << "f32_4 ";}
            else if (std::is_same<base_t, f32_8 >::value){ss << "f32_8 ";}
            else if (std::is_same<base_t, f32_16>::value){ss << "f32_16";}
            else if (std::is_same<base_t, f64   >::value){ss << "f64   ";}
            else if (std::is_same<base_t, f64_2 >::value){ss << "f64_2 ";}
            else if (std::is_same<base_t, f64_3 >::value){ss << "f64_3 ";}
            else if (std::is_same<base_t, f64_4 >::value){ss << "f64_4 ";}
            else if (std::is_same<base_t, f64_8 >::value){ss << "f64_8 ";}
            else if (std::is_same<base_t, f64_16>::value){ss << "f64_16";}
            else if (std::is_same<base_t, u32   >::value){ss << "u32   ";}
            else if (std::is_same<base_t, u64   >::value){ss << "u64   ";}
            else if (std::is_same<base_t, u32_3 >::value){ss << "u32_3 ";}
            else if (std::is_same<base_t, u64_3 >::value){ss << "u64_3 ";}
            else {
                ss << "unknown";
            }

            ss << "\n";

            index ++;
        });
    }

    }

    return ss.str();
}

std::vector<std::string> PatchDataLayout::get_field_names(){
    std::vector<std::string> ret ;

    for(var_t & v : fields){
        v.visit(
            [&](auto & field){
                ret.push_back(field.name);
            }
        );
    }

    return ret;
}


}