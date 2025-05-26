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
 * @file compute_neigh_graph.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::basegodunov::modules::details {

    /**
     * @brief Create a neighbour graph using a class that will list the ids of the found neighbourgh
     * NeighFindKernel will list the index and that function will run it twice to generate the graph
     *
     * @tparam NeighFindKernel the neigh find kernel
     * @tparam Args arguments that will be forwarded to the kernel
     * @param q the sycl queue
     * @param graph_nodes the number of graph nodes
     * @param args arguments that will be forwarded to the kernel
     * @return shammodels::basegodunov::modules::NeighGraph the neigh graph
     */
    template<class NeighFindKernel, class... Args>
    shammodels::basegodunov::modules::NeighGraph
    compute_neigh_graph(sycl::queue &q, u32 graph_nodes, Args &&...args) {

        // [i] is the number of link for block i in mpdat (last value is 0)
        sycl::buffer<u32> link_counts(graph_nodes + 1);

        // fill buffer with number of link in the block graph
        q.submit([&](sycl::handler &cgh) {
            NeighFindKernel ker(cgh, std::forward<Args>(args)...);
            sycl::accessor link_cnt{link_counts, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, graph_nodes, "count block graph link", [=](u64 gid) {
                u32 id_a              = (u32) gid;
                u32 block_found_count = 0;

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    block_found_count++;
                });

                link_cnt[id_a] = block_found_count;
            });
        });

        // set the last val to 0 so that the last slot after exclusive scan is the sum
        shamalgs::memory::set_element<u32>(q, link_counts, graph_nodes, 0);

        sycl::buffer<u32> link_cnt_offsets
            = shamalgs::numeric::exclusive_sum(q, link_counts, graph_nodes + 1);

        u32 link_cnt = shamalgs::memory::extract_element(q, link_cnt_offsets, graph_nodes);

        sycl::buffer<u32> ids_links(link_cnt);

        // find the neigh ids
        q.submit([&](sycl::handler &cgh) {
            NeighFindKernel ker(cgh, std::forward<Args>(args)...);
            sycl::accessor cnt_offsets{link_cnt_offsets, cgh, sycl::read_only};
            sycl::accessor links{ids_links, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, graph_nodes, "get ids block graph link", [=](u64 gid) {
                u32 id_a = (u32) gid;

                u32 next_link_idx = cnt_offsets[id_a];

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    links[next_link_idx] = id_b;
                    next_link_idx++;
                });
            });
        });

        using Graph = shammodels::basegodunov::modules::NeighGraph;
        return Graph(
            Graph{std::move(link_cnt_offsets), std::move(ids_links), link_cnt, graph_nodes});
    };

} // namespace shammodels::basegodunov::modules::details
