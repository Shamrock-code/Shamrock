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
 * @file AsciiSplitDump.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/string.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

/**
 * @brief A class to dump a simulation state into ASCII files
 * @todo add exemple of usage
 */
class AsciiSplitDump {

    /**
     * @brief A helper struct to hold a single file
     */
    struct PatchDump {
        std::ofstream file; ///< The file stream

        /**
         * @brief Open a file for writing
         *
         * @param id_patch the patch id
         * @param fileprefix the file prefix (without the patch id)
         */
        void open(u64 id_patch, std::string fileprefix) {
            const std::filesystem::path path{
                fileprefix + "patch_" + shambase::format("{:04d}", id_patch) + ".txt"};
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }

            file.open(path);
        }

        /**
         * @brief Change the table name in the file
         *
         * @param table_name the new table name
         * @param type the type of the table (e.g. "double")
         */
        void change_table_name(std::string table_name, std::string type) {
            file << "--> " + table_name + " " + "type=" + type + "\n";
        }

        /**
         * @brief Write a single value to the file
         *
         * @param val the value to write
         */
        template<class T>
        void write_val(T val);

        /**
         * @brief Write a table to the file
         *
         * @param buf the table to write
         * @param len the length of the table
         */
        template<class T>
        void write_table(std::vector<T> buf, u32 len);

        /**
         * @brief Write a table to the file
         *
         * @param buf the table to write
         * @param len the length of the table
         */
        template<class T>
        void write_table(sycl::buffer<T> buf, u32 len);

        /**
         * @brief Close the file
         */
        void close() { file.close(); }
    };

    /**
     * @brief A map from patch id to file
     *
     * This map is used to store the files associated with each patch.
     * The file is opened when the patch is added to the map, and closed
     * when the patch is removed from the map.
     */
    shambase::DistributedData<PatchDump> dump_dist;

    /**
     * @brief The file prefix for the patch files
     *
     * The file name for each patch is constructed by appending the patch id
     * to the file prefix.
     */
    std::string fileprefix;

    public:
    /**
     * @brief Get a reference to a file
     *
     * @param id the patch id
     * @return a reference to the file associated with the patch
     */
    inline PatchDump &get_file(u64 id) { return dump_dist.get(id); }

    /**
     * @brief Constructor
     *
     * @param fileprefix the file prefix (without the patch id)
     */
    explicit AsciiSplitDump(std::string fileprefix) : fileprefix(std::move(fileprefix)) {}

    /**
     * @brief Create a new file for the given patch
     *
     * @param id the patch id
     */
    inline void create_id(u64 id) { dump_dist.add_obj(id, {})->second.open(id, fileprefix); }
};
