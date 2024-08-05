// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file generic_opts.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file handler generic cli & env options
 *
 */

namespace shamcmdopt {

    /**
     * @brief Register generic cli and env variables options
     *
     */
    void register_cmdopt_generic_opts();

    /**
     * @brief Process generic cli and env variables options
     */
    void process_cmdopt_generic_opts();

    /**
     * @brief Print the help message.
     */
    void print_help();

} // namespace shamcmdopt
