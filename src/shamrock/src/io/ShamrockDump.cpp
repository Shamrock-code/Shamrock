// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ShamrockDump.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/io/ShamrockDump.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/logs.hpp"

namespace shamrock {

    void write_shamrock_dump(std::string fname, std::string metadata_user, PatchScheduler &sched) {
        std::string metadata_patch = sched.serialize_patch_metadata().dump(4);

        using namespace shamrock::patch;

        std::vector<u64> pids;
        std::vector<u64> bytecounts;
        std::vector<sham::DeviceBuffer<u8>> datas;

        // serialize patchdatas and push them into dat
        sched.patch_data.for_each_patchdata([&](u64 pid, PatchDataLayer &pdat) {
            auto ser_sz = pdat.serialize_buf_byte_size();
            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
            ser.allocate(ser_sz);
            pdat.serialize_buf(ser);

            auto tmp         = ser.finalize();
            size_t bytecount = tmp.get_size();

            pids.push_back(pid);
            bytecounts.push_back(bytecount);
            datas.push_back(std::move(tmp));
        });

        std::vector<u64> all_pids;
        std::vector<u64> all_bytecounts;

        shamalgs::collective::vector_allgatherv(
            pids, get_mpi_type<u64>(), all_pids, get_mpi_type<u64>(), MPI_COMM_WORLD);
        shamalgs::collective::vector_allgatherv(
            bytecounts, get_mpi_type<u64>(), all_bytecounts, get_mpi_type<u64>(), MPI_COMM_WORLD);

        std::vector<u64> all_offsets = all_bytecounts;

        std::exclusive_scan(all_offsets.begin(), all_offsets.end(), all_offsets.begin(), u64{0});

        using namespace nlohmann;

        json j;
        j["pids"]       = all_pids;
        j["bytecounts"] = all_bytecounts;
        j["offsets"]    = all_offsets;

        std::string sout = j.dump(4);

        // Write to the file

        u64 head_ptr = 0;
        MPI_File mfile{};

        shamcomm::open_reset_file(mfile, fname);

        shambase::Timer timer;
        timer.start();

        // do some perf investigation before enabling preallocation
        bool preallocate = false;
        if (preallocate) {
            MPI_Offset tot_byte = all_offsets.back() + all_bytecounts.back() + metadata_user.size()
                                  + metadata_patch.size() + sout.size() + sizeof(std::size_t) * 3;
            MPICHECK(MPI_File_preallocate(mfile, tot_byte));
        }

        shamalgs::collective::write_header(mfile, metadata_user, head_ptr);
        shamalgs::collective::write_header(mfile, metadata_patch, head_ptr);
        shamalgs::collective::write_header(mfile, sout, head_ptr);

        shamlog_debug_ln(
            "ShamrockDump",
            shambase::format(
                "table sizes {} {} {}", metadata_patch.size(), metadata_user.size(), sout.size()));

        if (/*do check*/ true) {
            auto check_same_mpi = [](std::string s) {
                u64 out = shamalgs::collective::allreduce_sum(s.size());
                if (out != s.size() * shamcomm::world_size()) {
                    logger::err_ln(
                        "ShamrockDump",
                        shambase::format(
                            "string size mismatch between all processes,\n    size : {}\nthe "
                            "string : {}\n",
                            s.size(),
                            s));
                    shambase::throw_with_loc<std::runtime_error>(
                        "size mismatch in shamrock dump header");
                }
            };

            check_same_mpi(metadata_user);
            check_same_mpi(metadata_patch);
            check_same_mpi(sout);
        }

        if (!shamcmdopt::getenv_str("SHAMDUMP_OFFSET_MODE_OLD").has_value()) {
            // reset MPI view
            MPICHECK(MPI_File_set_view(mfile, 0, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));
        }

        // map of patch id -> all_pids idx
        std::unordered_map<u64, size_t> map{};
        for (u32 i = 0; i < all_pids.size(); i++) {
            map[all_pids[i]] = i;
        }

        for (u32 i = 0; i < datas.size(); i++) {

            u64 pid       = pids[i];
            u64 bytecount = bytecounts[i];

            size_t off = all_offsets[map[pid]];
            auto &data = datas[i];

            shamcomm::CommunicationBuffer buf(data, shamsys::instance::get_compute_scheduler_ptr());

            shamalgs::collective::write_at<u8>(mfile, buf.get_ptr(), bytecount, head_ptr + off);
        }

        // write data to file

        MPI_File_close(&mfile);
        timer.end();

        if (shamcomm::world_rank() == 0) {
            size_t plist_len = all_offsets.size();
            size_t max_head = all_offsets[plist_len - 1] + all_bytecounts[plist_len - 1] + head_ptr;
            logger::info_ln(
                "Shamrock Dump",
                shambase::format(
                    "dump to {}\n              - took {}, bandwidth = {}/s",
                    fname,
                    timer.get_time_str(),
                    shambase::readable_sizeof(max_head / timer.elasped_sec())));
        }
    }

    void load_shamrock_dump(std::string fname, std::string &metadata_user, ShamrockCtx &ctx) {

        u64 head_ptr = 0;
        MPI_File mfile{};

        shamcomm::open_read_only_file(mfile, fname);

        shambase::Timer timer;
        timer.start();

        std::string metadata_patch{};
        std::string patchdata_infos{};

        metadata_user   = shamalgs::collective::read_header(mfile, head_ptr);
        metadata_patch  = shamalgs::collective::read_header(mfile, head_ptr);
        patchdata_infos = shamalgs::collective::read_header(mfile, head_ptr);

        if (!shamcmdopt::getenv_str("SHAMDUMP_OFFSET_MODE_OLD").has_value()) {
            // reset MPI view
            MPICHECK(MPI_File_set_view(mfile, 0, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));
        }
        // logger::raw_ln(metadata_user, metadata_patch, patchdata_infos);

        using namespace nlohmann;

        json jmeta_patch = json::parse(metadata_patch);
        json jpdat_info  = json::parse(patchdata_infos);

        ctx.pdata_layout_new();
        *ctx.pdl = jmeta_patch.at("patchdata_layout").get<patch::PatchDataLayerLayout>();
        ctx.init_sched(
            jmeta_patch.at("crit_patch_split").get<u64>(),
            jmeta_patch.at("crit_patch_merge").get<u64>());

        auto &sched = shambase::get_check_ref(ctx.sched);

        sched.patch_list = jmeta_patch.at("patchlist").get<SchedulerPatchList>();
        sched.patch_tree = jmeta_patch.at("patchtree").get<scheduler::PatchTree>();
        sched.patch_data.sim_box.from_json(jmeta_patch.at("sim_box"));

        // edit patch owner to fit in new world size, or spread if more processes now
        // a bit dirty but gets the job done for now
        // ideally we should call a load balance once
        for (auto &p : sched.patch_list.global) {
            p.node_owner_id = p.node_owner_id % shamcomm::world_size();
        }

        // rebuild local patch list
        auto loc_ids = sched.patch_list.build_local();

        // Load patchdata according to new LB

        std::vector<u64> all_offsets;
        std::vector<u64> all_pids;
        std::vector<u64> all_bytecounts;

        all_bytecounts = jpdat_info.at("bytecounts").get<std::vector<u64>>();
        all_offsets    = jpdat_info.at("offsets").get<std::vector<u64>>();
        all_pids       = jpdat_info.at("pids").get<std::vector<u64>>();

        struct PatchFileOffset {
            u64 offset, bytecount;
        };

        std::unordered_map<u64, PatchFileOffset> off_table;

        for (u32 i = 0; i < all_bytecounts.size(); i++) {
            off_table[all_pids[i]] = {all_offsets[i], all_bytecounts[i]};
        }

        for (const auto &p : sched.patch_list.local) {
            u64 pid            = p.id_patch;
            auto loc_file_info = off_table[pid];

            shamcomm::CommunicationBuffer buf(
                loc_file_info.bytecount, shamsys::instance::get_compute_scheduler_ptr());

            shamalgs::collective::read_at<u8>(
                mfile, buf.get_ptr(), loc_file_info.bytecount, head_ptr + loc_file_info.offset);

            sham::DeviceBuffer<u8> out = shamcomm::CommunicationBuffer::convert_usm(std::move(buf));

            shamalgs::SerializeHelper ser(
                shamsys::instance::get_compute_scheduler_ptr(), std::move(out));

            patch::PatchDataLayer pdat
                = patch::PatchDataLayer::deserialize_buf(ser, shambase::get_check_ref(ctx.pdl));

            sched.patch_data.owned_data.add_obj(pid, std::move(pdat));
        }

        MPI_File_close(&mfile);
        timer.end();

        if (shamcomm::world_rank() == 0) {
            size_t plist_len = all_offsets.size();
            size_t max_head = all_offsets[plist_len - 1] + all_bytecounts[plist_len - 1] + head_ptr;
            logger::info_ln(
                "Shamrock Dump",
                shambase::format(
                    "load dump from {}\n              - took {}, bandwidth = {}/s",
                    fname,
                    timer.get_time_str(),
                    shambase::readable_sizeof(max_head / timer.elasped_sec())));
        }
    }

} // namespace shamrock
