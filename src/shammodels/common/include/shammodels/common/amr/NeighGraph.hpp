// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NeighGraph.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"

namespace shammodels::basegodunov::modules {

    struct NeighGraphLinkiterator;

    struct NeighGraph {
        sham::DeviceBuffer<u32> node_link_offset;
        sham::DeviceBuffer<u32> node_links;
        u32 link_count;
        u32 obj_cnt;

        std::optional<sham::DeviceBuffer<u32>> antecedent = std::nullopt;

        void compute_antecedent(sham::DeviceScheduler_ptr &dev_sched) {
            sham::DeviceBuffer<u32> ret(link_count, dev_sched);

            auto &q = dev_sched->get_queue(0);
            sham::EventList deps;

            sham::kernel_call(
                q,
                sham::MultiRef{node_link_offset},
                sham::MultiRef{ret},
                obj_cnt,
                [](u32 gid, const u32 *__restrict offset, u32 *__restrict ante) {
                    u32 min_ids = offset[gid];
                    u32 max_ids = offset[gid + 1];
                    for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                        ante[id_s] = gid;
                    }
                });

            antecedent = std::move(ret);
        }

        struct ro_access {

            const u32 *node_link_offset;
            const u32 *node_links;

            template<class Functor_iter>
            inline void for_each_object_link(const u32 &cell_id, Functor_iter &&func_it) const {
                u32 min_ids = node_link_offset[cell_id];
                u32 max_ids = node_link_offset[cell_id + 1];
                for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                    func_it(node_links[id_s]);
                }
            }

            template<class Functor_iter>
            inline void for_each_object_link_id(const u32 &cell_id, Functor_iter &&func_it) const {
                u32 min_ids = node_link_offset[cell_id];
                u32 max_ids = node_link_offset[cell_id + 1];
                for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                    func_it(node_links[id_s], id_s);
                }
            }

            template<class Functor_iter>
            inline u32 for_each_object_link_cnt(const u32 &cell_id, Functor_iter &&func_it) const {
                u32 min_ids = node_link_offset[cell_id];
                u32 max_ids = node_link_offset[cell_id + 1];
                for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                    func_it(node_links[id_s]);
                }
                return max_ids - min_ids;
            }
        };

        ro_access get_read_access(sham::EventList &e) {
            return ro_access{node_link_offset.get_read_access(e), node_links.get_read_access(e)};
        }

        void complete_event_state(sycl::event &e) {
            node_link_offset.complete_event_state(e);
            node_links.complete_event_state(e);
        }
    };

    using AMRGraph = NeighGraph;

    enum Direction {
        xp = 0,
        xm = 1,
        yp = 2,
        ym = 3,
        zp = 4,
        zm = 5,
    };

    template<class Tvec, class TgridVec>
    struct OrientedAMRGraph {

        const std::array<TgridVec, 6> offset_check{
            TgridVec{1, 0, 0},
            TgridVec{-1, 0, 0},
            TgridVec{0, 1, 0},
            TgridVec{0, -1, 0},
            TgridVec{0, 0, 1},
            TgridVec{0, 0, -1},
        };

        std::array<std::unique_ptr<AMRGraph>, 6> graph_links;
    };
} // namespace shammodels::basegodunov::modules
