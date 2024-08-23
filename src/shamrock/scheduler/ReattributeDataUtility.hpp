// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ReattributeDataUtility.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/string.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <vector>

namespace shamrock {
    class ReattributeDataUtility {
        PatchScheduler &sched;

        public:
        ReattributeDataUtility(PatchScheduler &sched) : sched(sched) {}

        template<class T>
        shambase::DistributedData<sycl::buffer<u64>>
        compute_new_pid(SerialPatchTree<T> &sptree, u32 ipos) {

            StackEntry stack_loc{};

            shambase::DistributedData<sycl::buffer<u64>> newid_buf_map;

            sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
                if (!pdat.is_empty()) {

                    PatchDataField<T> &pos_field = pdat.get_field<T>(ipos);

                    newid_buf_map.add_obj(
                        id,
                        sptree.compute_patch_owner(
                            shamsys::instance::get_compute_queue(),
                            shambase::get_check_ref(pos_field.get_buf()),
                            pos_field.size()));

                    bool err_id_in_newid = false;
                    {
                        sycl::host_accessor nid{newid_buf_map.get(id), sycl::read_only};
                        for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                            bool err        = nid[i] == u64_max;
                            err_id_in_newid = err_id_in_newid || (err);
                        }
                    }

                    if (err_id_in_newid) {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "a new id could not be computed");
                    }
                }
            });

            return newid_buf_map;
        }

        inline shambase::DistributedDataShared<shamrock::patch::PatchData>
        extract_elements(shambase::DistributedData<sycl::buffer<u64>> new_pid) {
            shambase::DistributedDataShared<patch::PatchData> part_exchange;

            StackEntry stack_loc{};

            using namespace shamrock::patch;

            std::unordered_map<u64, u64> histogram_extract;

            sched.patch_data.for_each_patchdata([&](u64 current_pid,
                                                    shamrock::patch::PatchData &pdat) {
                histogram_extract[current_pid] = 0;
                if (!pdat.is_empty()) {

                    sycl::host_accessor nid{new_pid.get(current_pid), sycl::read_only};

                    if (false) {

                        const u32 cnt = pdat.get_obj_cnt();

                        for (u32 i = cnt - 1; i < cnt; i--) {
                            u64 new_pid = nid[i];
                            if (current_pid != new_pid) {

                                if (!part_exchange.has_key(current_pid, new_pid)) {
                                    part_exchange.add_obj(
                                        current_pid, new_pid, PatchData(sched.pdl));
                                }

                                part_exchange.for_each(
                                    [&](u64 _old_id, u64 _new_id, PatchData &pdat_int) {
                                        if (_old_id == current_pid && _new_id == new_pid) {
                                            pdat.extract_element(i, pdat_int);
                                            histogram_extract[current_pid]++;
                                        }
                                    });
                            }
                        }
                    } else {
                        std::vector<u32> keep_ids;
                        std::unordered_map<u64, std::vector<u32>> extract_indexes;

                        const u32 cnt = pdat.get_obj_cnt();
                        for (u32 i = 0; i < cnt; i++) {
                            u64 new_pid = nid[i];
                            if (current_pid != new_pid) {
                                extract_indexes[new_pid].push_back(i);
                                histogram_extract[current_pid]++;
                            } else {
                                keep_ids.push_back(i);
                            }
                        }

                        for (auto &[new_id, vec] : extract_indexes) {

                            u64 new_pid                   = new_id;
                            std::vector<u32> &idx_extract = vec;

                            if (!part_exchange.has_key(current_pid, new_pid)) {
                                part_exchange.add_obj(current_pid, new_pid, PatchData(sched.pdl));
                            }

                            part_exchange.for_each(
                                [&](u64 _old_id, u64 _new_id, PatchData &pdat_int) {
                                    if (_old_id == current_pid && _new_id == new_pid) {
                                        pdat.append_subset_to(idx_extract, pdat_int);
                                    }
                                });
                        }

                        sycl::buffer<u32> keep_idx = shamalgs::memory::vec_to_buf(keep_ids);
                        pdat.keep_ids(keep_idx, keep_ids.size());
                    }
                }
            });

            for (auto &[k, v] : histogram_extract) {
                logger::debug_ln("ReattributeDataUtility", "patch", k, "extract=", v);
            }

            return part_exchange;
        }

        template<class T>
        inline void
        reatribute_patch_objects(SerialPatchTree<T> &sptree, std::string position_field) {
            StackEntry stack_loc{};

            using namespace shambase;
            using namespace shamrock::patch;

            u32 ipos = sched.pdl.get_field_idx<T>(position_field);

            DistributedData<sycl::buffer<u64>> new_pid = compute_new_pid(sptree, ipos);

            DistributedDataShared<patch::PatchData> part_exchange = extract_elements(new_pid);
            DistributedDataShared<patch::PatchData> recv_dat;

            shamalgs::collective::serialize_sparse_comm<PatchData>(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::move(part_exchange),
                recv_dat,
                [&](u64 id) {
                    return sched.get_patch_rank_owner(id);
                },
                [](PatchData &pdat) {
                    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
                    ser.allocate(pdat.serialize_buf_byte_size());
                    pdat.serialize_buf(ser);
                    return ser.finalize();
                },
                [&](std::unique_ptr<sycl::buffer<u8>> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<std::unique_ptr<sycl::buffer<u8>>>(buf));
                    return PatchData::deserialize_buf(ser, sched.pdl);
                });

            recv_dat.for_each([&](u64 sender, u64 receiver, PatchData &pdat) {
                logger::debug_ln("Part Exchanges", format("send = {} recv = {}", sender, receiver));
                sched.patch_data.get_pdat(receiver).insert_elements(pdat);
            });
        }
    };
} // namespace shamrock
