// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file LegacyVtkWritter.hpp
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

    inline void
    write_shamrock_dump(std::string fname, std::string metadata_user, PatchScheduler &sched) {
        std::string metadata_patch = sched.serialize_patch_metadata().dump(4);

        using namespace shamrock::patch;

        std::vector<u64> pids;
        std::vector<size_t> bytecounts;
        std::vector<std::unique_ptr<sycl::buffer<u8>>> datas;

        // serialize patchdatas and push them into dat
        sched.patch_data.for_each_patchdata([&](u64 pid, PatchData &pdat) {
            auto ser_sz = pdat.serialize_buf_byte_size();
            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
            ser.allocate(ser_sz);
            pdat.serialize_buf(ser);

            auto tmp         = ser.finalize();
            size_t bytecount = shambase::get_check_ref(tmp).size();

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

        std::exclusive_scan(all_offsets.begin(), all_offsets.end(), all_offsets.begin(), 0);

        using namespace nlohmann;

        json j;
        j["pids"]       = all_pids;
        j["bytecounts"] = all_bytecounts;
        j["offsets"]    = all_offsets;

        std::string sout = j.dump(4);

        // Write to the file

        size_t head_ptr = 0;
        MPI_File mfile{};

        shamcomm::open_reset_file(mfile, fname);

        shamalgs::collective::write_header(mfile, metadata_user, head_ptr);
        shamalgs::collective::write_header(mfile, metadata_patch, head_ptr);
        shamalgs::collective::write_header(mfile, sout, head_ptr);

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

            shamcomm::CommunicationBuffer buf(
                shambase::get_check_ref(data), shamsys::instance::get_compute_scheduler());

            shamalgs::collective::write_at<u8>(mfile, buf.get_ptr(), bytecount, head_ptr + off);
        }

        // write data to file

        MPI_File_close(&mfile);
    }

    namespace details {
        
    }

    inline void load_shamrock_dump(std::string fname, std::string & metadata_user, ShamrockCtx & ctx){

        size_t head_ptr = 0;
        MPI_File mfile{};

        shamcomm::open_read_only_file(mfile, fname);

        std::string metadata_patch {};
        std::string patchdata_infos {};

        metadata_user = shamalgs::collective::read_header(mfile, head_ptr);
        metadata_patch = shamalgs::collective::read_header(mfile, head_ptr);
        patchdata_infos = shamalgs::collective::read_header(mfile, head_ptr);

        //logger::raw_ln(metadata_user, metadata_patch, patchdata_infos);

        using namespace nlohmann;

        json jmeta_patch = json::parse(metadata_patch);
        json jpdat_info = json::parse(patchdata_infos);

        ctx.pdata_layout_new();
        * ctx.pdl = jmeta_patch.at("patchdata_layout").get<patch::PatchDataLayout>();
        ctx.init_sched(jmeta_patch.at("crit_patch_split").get<u64>(), jmeta_patch.at("crit_patch_merge").get<u64>());

        auto & sched = shambase::get_check_ref(ctx.sched);

        sched.patch_list = jmeta_patch.at("patchlist").get<SchedulerPatchList>();
        sched.patch_tree = jmeta_patch.at("patchtree").get<scheduler::PatchTree>();
        sched.patch_data.sim_box.from_json(jmeta_patch.at("sim_box"));

        // edit patch owner to fit in new world size, or spread if more processes now
        // a bit dirty but gets the job done for now
        // ideally we should call a load balance once
        for (auto & p : sched.patch_list.global) {
            p.node_owner_id = p.node_owner_id % shamcomm::world_size();
        }

        // rebuild local patch list
        auto loc_ids = sched.patch_list.build_local();
        
        // Load patchdata according to new LB

        std::vector<u64> all_offsets;
        std::vector<u64> all_pids;
        std::vector<u64> all_bytecounts;

        all_bytecounts = jpdat_info.at("bytecounts").get<std::vector<u64>>();
        all_offsets = jpdat_info.at("offsets").get<std::vector<u64>>();
        all_pids = jpdat_info.at("pids").get<std::vector<u64>>();

        struct PatchFileOffset{
            u64 offset, bytecount;
        };

        std::unordered_map<u64, PatchFileOffset> off_table;

        for(u32 i = 0; i < all_bytecounts.size(); i++){
            off_table[all_pids[i]] = {all_offsets[i], all_bytecounts[i]};
        }

        for (const auto & p : sched.patch_list.local) {
            u64 pid = p.id_patch;
            auto loc_file_info = off_table[pid];

            shamcomm::CommunicationBuffer buf(loc_file_info.bytecount, shamsys::instance::get_compute_scheduler());

            shamalgs::collective::read_at<u8>(mfile, buf.get_ptr(), loc_file_info.bytecount, head_ptr + loc_file_info.offset);

            std::unique_ptr<sycl::buffer<u8>> out = std::make_unique<sycl::buffer<u8>> (shamcomm::CommunicationBuffer::convert(std::move(buf)));

            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr(), std::move(out));

            patch::PatchData pdat = patch::PatchData::deserialize_buf(ser, shambase::get_check_ref(ctx.pdl));

            sched.patch_data.owned_data.add_obj(pid, std::move(pdat));

        }

        MPI_File_close(&mfile);
    }

} // namespace shamrock
