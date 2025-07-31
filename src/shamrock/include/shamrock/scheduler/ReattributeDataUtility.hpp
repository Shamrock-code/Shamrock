// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ReattributeDataUtility.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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

    /**
     * @brief Utility class used to move the objects between patches.
     *
     * The class is used to recompute the ownership of the objects in the patches
     * based on their position in space.
     *
     */
    class ReattributeDataUtility {
        PatchScheduler &sched; ///< Scheduler to bind onto

        public:
        /**
         * @brief Constructor
         *
         * @param sched The PatchScheduler to work on.
         */
        ReattributeDataUtility(PatchScheduler &sched) : sched(sched) {}

        /**
         * @brief Computes the new patch owner IDs for the objects in the patches based on their
         * position in space.
         *
         * @param sptree The SerialPatchTree used to compute the patch owners.
         * @param ipos The index of the position field in the PatchData.
         *
         * @return A DistributedData containing the new patch IDs for each patch.
         *
         * @throws std::runtime_error If a new ID could not be computed for an object (out of
         * bound).
         */
        template<class T>
        shambase::DistributedData<sycl::buffer<u64>>
        compute_new_pid(SerialPatchTree<T> &sptree, u32 ipos) {

            StackEntry stack_loc{};

            shambase::DistributedData<sycl::buffer<u64>> newid_buf_map;

            sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData &pdat) {
                if (!pdat.is_empty()) {

                    PatchDataField<T> &pos_field = pdat.get_field<T>(ipos);

                    if (pos_field.get_nvar() != 1) {
                        shambase::throw_unimplemented();
                    }

                    newid_buf_map.add_obj(
                        id,
                        sptree.compute_patch_owner(
                            shamsys::instance::get_compute_scheduler_ptr(),
                            pos_field.get_buf(),
                            pos_field.get_obj_cnt()));

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

        /**
         * @brief Extracts elements that do not belong to a patch from the patch data based on the
         * new patch IDs.
         *
         * This function iterates over the patch data and extracts elements that need to be moved to
         * a different patch. It uses the new patch IDs to determine which elements to extract and
         * where to move them.
         *
         * @param new_pid A distributed data object containing the new patch IDs.
         *
         * @return A shared distributed data object containing the extracted patch data.
         */
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
                shamlog_debug_ln("ReattributeDataUtility", "patch", k, "extract=", v);
            }

            return part_exchange;
        }

        /**
         * @brief Reattribute objects based on a given position field.
         *
         * This function computes new patch IDs for each object in the PatchData,
         * extracts elements to be exchanged between patches, and then updates the patch data
         * with the received elements.
         *
         * @param sptree the SerialPatchTree
         * @param position_field the name of the main field used to determine the new patch IDs
         */
        template<class T>
        inline void
        reatribute_patch_objects(SerialPatchTree<T> &sptree, std::string position_field) {
            StackEntry stack_loc{};

            using namespace shambase;
            using namespace shamrock::patch;

            u32 ipos = sched.pdl.get_field_idx<T>(position_field);

            DistributedData<sycl::buffer<u64>> new_pid = compute_new_pid(sptree, ipos);

            DistributedDataShared<patch::PatchData> part_exchange = extract_elements(new_pid);

            part_exchange.for_each([](u64 sender, u64 receiver, PatchData &pdat) {
                shamlog_debug_ln("ReattributeDataUtility", sender, receiver, pdat.get_obj_cnt());
            });

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
                [&](sham::DeviceBuffer<u8> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<sham::DeviceBuffer<u8>>(buf));
                    return PatchData::deserialize_buf(ser, sched.pdl);
                });

            recv_dat.for_each([&](u64 sender, u64 receiver, PatchData &pdat) {
                shamlog_debug_ln("Part Exchanges", format("send = {} recv = {}", sender, receiver));
                sched.patch_data.get_pdat(receiver).insert_elements(pdat);
            });
        }
    };

} // namespace shamrock
