// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file endian.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/integer.hpp"
#include "shambackends/typeAliasVec.hpp"

namespace shambase {


    /**
     * @brief check if the cpu is in little endian
     * p. 45 of Pointers in C
     * 
     * @return true 
     * @return false 
     */
    inline bool is_little_endian() {
        short int word = 0x0001;
        char *byte     = (char *)&word;
        return (byte[0] ? 1 : 0);
    }

    /**
     * @brief swap the endiannes of the value a
     * 
     * @tparam T 
     * @param a 
     */
    template<class T>
    inline void endian_swap(T &a) {

        constexpr i32 sz = sizeof(a);

        auto constexpr lambd = []() {
            if constexpr (sz % 2 == 0) {
                return sz / 2;
            } else {
                return (sz - 1) / 2;
            }
        };

        constexpr i32 steps = lambd();

        u8 *bytes = (u8 *)&a;

        for (i32 i = 0; i < steps; i++) {
            xor_swap(bytes[i], bytes[sz - 1 - i]);
        }
    }

    /**
     * @brief return the input value with swapped endiannes
     * 
     * @tparam T 
     * @param a 
     * @return T 
     */
    template<class T> 
    inline T get_endian_swap(T a){
        T ret = a;
        endian_swap(ret);
        return ret;
    }
    

} // namespace shambase