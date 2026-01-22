// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MpiDataTypeHandler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamsys/legacy/log.hpp"
#include <functional>
#include <string>

namespace shamsys::mpidtypehandler {

    using fct_sig = std::function<void()>;

    inline std::vector<fct_sig> static_init_shamrock_mpidtype{};
    inline std::vector<fct_sig> static_free_shamrock_mpidtype{};

    inline std::vector<std::string> static_init_shamrock_mpidtype_names{};
    inline std::vector<std::string> static_free_shamrock_mpidtype_names{};

    inline void init_mpidtype() {
        for (u32 i = 0; i < static_init_shamrock_mpidtype.size(); i++) {
            auto fct  = static_init_shamrock_mpidtype[i];
            auto name = static_init_shamrock_mpidtype_names[i];
            shamlog_debug_mpi_ln("MpiDTypehandler", "initialising type :", name);
            fct();
        }
    }

    inline void free_mpidtype() {
        for (u32 i = 0; i < static_free_shamrock_mpidtype.size(); i++) {
            auto fct  = static_free_shamrock_mpidtype[i];
            auto name = static_free_shamrock_mpidtype_names[i];
            shamlog_debug_mpi_ln("MpiDTypehandler", "freeing type :", name);
            fct();
        }
    }

    struct MPIDTypeinit {
        inline explicit MPIDTypeinit(fct_sig t, std::string name) {
            static_init_shamrock_mpidtype.push_back(std::move(t));
            static_init_shamrock_mpidtype_names.push_back(name);
        }
    };

    struct MPIDTypefree {
        inline explicit MPIDTypefree(fct_sig t, std::string name) {
            static_free_shamrock_mpidtype.push_back(std::move(t));
            static_free_shamrock_mpidtype_names.push_back(name);
        }
    };

} // namespace shamsys::mpidtypehandler

/**
 * @brief register a static init function to initialize a mpi type
 *
 */
#define Register_MPIDtypeInit(placeholdername, name)                                               \
    void mpiinit_##placeholdername();                                                              \
    void (*mpiinit_ptr_##placeholdername)() = mpiinit_##placeholdername;                           \
    shamsys::mpidtypehandler::MPIDTypeinit mpiinit_class_obj_##placeholdername(                    \
        mpiinit_ptr_##placeholdername, name);                                                      \
    void mpiinit_##placeholdername()

/**
 * @brief register a static init function to free a mpi type
 *
 */
#define Register_MPIDtypeFree(placeholdername, name)                                               \
    void mpifree_##placeholdername();                                                              \
    void (*mpifree_ptr_##placeholdername)() = mpifree_##placeholdername;                           \
    shamsys::mpidtypehandler::MPIDTypefree mpifree_class_obj_##placeholdername(                    \
        mpifree_ptr_##placeholdername, name);                                                      \
    void mpifree_##placeholdername()
