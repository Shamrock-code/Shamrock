// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AsciiSplitDump.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/DistributedData.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl.hpp"
#include "shambase/type_aliases.hpp"
#include "shambase/sycl_vec_aliases.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

class AsciiSplitDump {

    struct PatchDump {
        std::ofstream file;

        void open(u64 id_patch, std::string fileprefix) {
            const std::filesystem::path path{
                fileprefix + "patch_" + shambase::format("{:04d}", id_patch) + ".txt"};
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }

            file.open(path);
        }

        void change_table_name(std::string table_name, std::string type) {
            file << "--> " + table_name + " " + "type=" + type + "\n";
        }

        template<class T>
        void write_val(T val);

        template<class T>
        void write_table(std::vector<T> buf, u32 len);

        template<class T>
        void write_table(sycl::buffer<T> buf, u32 len);

        void close() { file.close(); }
    };

    shambase::DistributedData<PatchDump> dump_dist;
    std::string fileprefix;

    public:
    inline PatchDump &get_file(u64 id) { return dump_dist.get(id); }

    explicit AsciiSplitDump(std::string fileprefix) : fileprefix(std::move(fileprefix)) {}

    inline void create_id(u64 id) { dump_dist.add_obj(id, {})->second.open(id, fileprefix); }
};