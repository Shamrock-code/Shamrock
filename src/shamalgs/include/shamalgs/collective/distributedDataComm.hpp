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
 * @file distributedDataComm.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/DistributedDataShared.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/sparseXchg.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include <functional>
#include <mpi.h>
#include <optional>
#include <stdexcept>
#include <vector>

namespace shamalgs::collective {

    using SerializedDDataComm = shambase::DistributedDataShared<sham::DeviceBuffer<u8>>;

    template<sham::USMKindTarget target>
    struct DDSCommCacheTarget {
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> cache1;
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> cache2;

        void set_sizes(
            sham::DeviceScheduler_ptr dev_sched,
            const std::vector<size_t> &sizes_cache1,
            const std::vector<size_t> &sizes_cache2) {

            __shamrock_stack_entry();

            // ensure correct length
            cache1.resize(sizes_cache1.size());
            cache2.resize(sizes_cache2.size());

            // if size is different, resize
            for (size_t i = 0; i < sizes_cache1.size(); i++) {
                if (cache1[i]) {
                    cache1[i]->resize(sizes_cache1[i], false);
                } else {
                    cache1[i] = std::make_unique<sham::DeviceBuffer<u8, target>>(
                        sizes_cache1[i], dev_sched);
                }
            }
            for (size_t i = 0; i < sizes_cache2.size(); i++) {
                if (cache2[i]) {
                    cache2[i]->resize(sizes_cache2[i], false);
                } else {
                    cache2[i] = std::make_unique<sham::DeviceBuffer<u8, target>>(
                        sizes_cache2[i], dev_sched);
                }
            }
        }

        inline void send_cache_write_buf_at(
            size_t buf_id, size_t offset, const sham::DeviceBuffer<u8> &buf) {
            buf.copy_range_offset(
                0, buf.get_size(), shambase::get_check_ref(cache1[buf_id]), offset);
        }

        inline void send_cache_read_buf_at(
            size_t buf_id, size_t offset, size_t size, sham::DeviceBuffer<u8> &buf) {
            buf.resize(size);
            shambase::get_check_ref(cache1[buf_id]).copy_range(offset, offset + size, buf);
        }

        inline void recv_cache_write_buf_at(
            size_t buf_id, size_t offset, const sham::DeviceBuffer<u8> &buf) {
            buf.copy_range_offset(
                0, buf.get_size(), shambase::get_check_ref(cache2[buf_id]), offset);
        }

        inline void recv_cache_read_buf_at(
            size_t buf_id, size_t offset, size_t size, sham::DeviceBuffer<u8> &buf) {
            buf.resize(size);
            shambase::get_check_ref(cache2[buf_id]).copy_range(offset, offset + size, buf);
        }
    };

    struct DDSCommCache {
        std::variant<DDSCommCacheTarget<sham::device>, DDSCommCacheTarget<sham::host>> cache;

        template<sham::USMKindTarget target>
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &get_cache1() {
            return shambase::get_check_ref(std::get_if<DDSCommCacheTarget<target>>(&cache)).cache1;
        }

        template<sham::USMKindTarget target>
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &get_cache2() {
            return shambase::get_check_ref(std::get_if<DDSCommCacheTarget<target>>(&cache)).cache2;
        }

        template<sham::USMKindTarget target>
        void set_sizes(
            sham::DeviceScheduler_ptr dev_sched,
            const std::vector<size_t> &sizes_cache1,
            const std::vector<size_t> &sizes_cache2) {

            __shamrock_stack_entry();

            // init if not there
            if (std::get_if<DDSCommCacheTarget<target>>(&cache) == nullptr) {
                cache = DDSCommCacheTarget<target>{};
            }

            std::get<DDSCommCacheTarget<target>>(cache).set_sizes(
                dev_sched, sizes_cache1, sizes_cache2);
        }

        inline void send_cache_write_buf_at(
            size_t buf_id, size_t offset, const sham::DeviceBuffer<u8> &buf) {
            std::visit(
                [&](auto &cache) {
                    cache.send_cache_write_buf_at(buf_id, offset, buf);
                },
                cache);
        }

        inline void send_cache_read_buf_at(
            size_t buf_id, size_t offset, size_t size, sham::DeviceBuffer<u8> &buf) {
            std::visit(
                [&](auto &cache) {
                    cache.send_cache_read_buf_at(buf_id, offset, size, buf);
                },
                cache);
        }

        inline void recv_cache_write_buf_at(
            size_t buf_id, size_t offset, const sham::DeviceBuffer<u8> &buf) {
            std::visit(
                [&](auto &cache) {
                    cache.recv_cache_write_buf_at(buf_id, offset, buf);
                },
                cache);
        }

        inline void recv_cache_read_buf_at(
            size_t buf_id, size_t offset, size_t size, sham::DeviceBuffer<u8> &buf) {
            std::visit(
                [&](auto &cache) {
                    cache.recv_cache_read_buf_at(buf_id, offset, size, buf);
                },
                cache);
        }
    };

    void distributed_data_sparse_comm(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        SerializedDDataComm &send_ddistrib_data,
        SerializedDDataComm &recv_distrib_data,
        std::function<i32(u64)> rank_getter,
        DDSCommCache &cache,
        std::optional<SparseCommTable> comm_table = {},
        size_t max_comm_size                      = i32_max - 1); // MPI msg size limit

    template<class T>
    inline void serialize_sparse_comm(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        shambase::DistributedDataShared<T> &&send_distrib_data,
        shambase::DistributedDataShared<T> &recv_distrib_data,
        std::function<i32(u64)> rank_getter,
        std::function<sham::DeviceBuffer<u8>(T &)> serialize,
        std::function<T(sham::DeviceBuffer<u8> &&)> deserialize,
        DDSCommCache &cache,
        std::optional<SparseCommTable> comm_table = {}) {

        StackEntry stack_loc{};

        shambase::DistributedDataShared<T> same_rank_tmp;
        // allow move op for same rank
        send_distrib_data.tranfer_all(
            [&](u64 l, u64 r) {
                return rank_getter(l) == rank_getter(r);
            },
            same_rank_tmp);

        SerializedDDataComm dcomm_send
            = send_distrib_data.template map<sham::DeviceBuffer<u8>>([&](u64, u64, T &obj) {
                  return serialize(obj);
              });

        SerializedDDataComm dcomm_recv;

        distributed_data_sparse_comm(dev_sched, dcomm_send, dcomm_recv, rank_getter, cache);

        recv_distrib_data = dcomm_recv.map<T>([&](u64, u64, sham::DeviceBuffer<u8> &buf) {
            // exchange the buffer held by the distrib data and give it to the deserializer
            return deserialize(std::move(buf));
        });

        shamlog_debug_ln(
            "SparseComm", "skipped", same_rank_tmp.get_native().size(), "communications");

        same_rank_tmp.tranfer_all(
            [&](u64 l, u64 r) {
                return true;
            },
            recv_distrib_data);
    }

    /**
     * @brief global ids = allgatherv(local_ids)
     *
     * @tparam T
     * @param src
     * @param local_ids
     * @param global_ids
     * @return shambase::DistributedData<T>
     */
    template<class T, class P>
    shambase::DistributedData<T> fetch_all_simple(
        shambase::DistributedData<T> &src,
        std::vector<P> local_ids,
        std::vector<P> global_ids,
        std::function<u64(P)> id_getter) {
        std::vector<T> vec_local(local_ids.size());
        for (u32 i = 0; i < local_ids.size(); i++) {
            vec_local[i] = src.get(id_getter(local_ids[i]));
        }

        std::vector<T> vec_global;
        vector_allgatherv(
            vec_local, get_mpi_type<T>(), vec_global, get_mpi_type<T>(), MPI_COMM_WORLD);

        shambase::DistributedData<T> ret;
        for (u32 i = 0; i < global_ids.size(); i++) {
            ret.add_obj(id_getter(global_ids[i]), T(vec_global[i]));
        }
        return ret;
    }

    /**
     * @brief global ids = allgatherv(local_ids)
     *
     * @tparam T
     * @param src
     * @param local_ids
     * @param global_ids
     * @return shambase::DistributedData<T>
     */
    template<class T, class P>
    shambase::DistributedData<T> fetch_all_storeload(
        shambase::DistributedData<T> &src,
        std::vector<P> local_ids,
        std::vector<P> global_ids,
        std::function<u64(P)> id_getter) {

        using Trepr          = typename T::Tload_store_repr;
        constexpr u32 reprsz = T::sz_load_store_repr;

        std::vector<T> vec_local(local_ids.size() * reprsz);
        for (u32 i = 0; i < local_ids.size(); i++) {
            src.get(id_getter(local_ids[i])).store(i * reprsz, vec_local);
        }

        std::vector<T> vec_global;
        vector_allgatherv(
            vec_local, get_mpi_type<T>(), vec_global, get_mpi_type<T>(), MPI_COMM_WORLD);

        shambase::DistributedData<T> ret;
        for (u32 i = 0; i < global_ids.size(); i++) {
            T tmp = T::load(i * reprsz, vec_global);
            ret.add_obj(id_getter(global_ids[i]), std::move(tmp));
        }
        return ret;
    }

} // namespace shamalgs::collective
