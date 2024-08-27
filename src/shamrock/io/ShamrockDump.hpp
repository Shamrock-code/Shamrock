// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ShamrockDump.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/collective/io.hpp"
#include "shamcomm/io.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <string>

namespace shamrock {

    /**
     * @brief Write a Shamrock dump file containing the current state of the patches and user
     * supplied metadata
     *
     * @todo Do some perf investigation before enabling preallocation
     *
     * @param fname The file name to write to
     * @param metadata_user The user-provided metadata to add to the dump
     * @param sched The patch scheduler to dump
     */
    void write_shamrock_dump(std::string fname, std::string metadata_user, PatchScheduler &sched);

    /**
     * @brief Load a Shamrock dump file and restore the state of the patches and retreive user
     * metadata
     * @param fname The file name to read from
     * @param metadata_user The user-provided metadata to store
     * @param ctx The Shamrock context to restore
     */
    void load_shamrock_dump(std::string fname, std::string &metadata_user, ShamrockCtx &ctx);

} // namespace shamrock
