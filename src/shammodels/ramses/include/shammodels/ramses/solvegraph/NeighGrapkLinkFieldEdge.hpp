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
 * @file NeighGrapkLinkFieldEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/common/amr/NeighGraphLinkField.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtree/RadixTree.hpp"
#include <functional>

namespace shammodels::basegodunov::solvergraph {

    template<class T>
    class NeighGrapkLinkFieldEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        u32 nvar;
        shambase::DistributedData<modules::NeighGraphLinkField<T>> link_fields;

        NeighGrapkLinkFieldEdge(std::string name, std::string texsymbol, u32 nvar)
            : IEdgeNamed(name, texsymbol), nvar(nvar) {}

        inline void
        check_size(const shambase::DistributedData<std::reference_wrapper<modules::AMRGraph>>
                       &graph) const {

            on_distributeddata_diff(
                link_fields,
                graph,
                [&](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing link field in distributed data at id " + std::to_string(id));
                },
                [&](u64 id) {
                    if (link_fields.get(id).link_count != graph.get(id).get().link_count) {
                        shambase::throw_with_loc<std::runtime_error>(
                            "Wrong size for link field in distributed data at id "
                            + std::to_string(id));
                    }
                },
                [&](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra link field in distributed data at id " + std::to_string(id));
                });
        }

        inline void resize_according_to(
            const shambase::DistributedData<std::reference_wrapper<modules::AMRGraph>> &graph) {

            on_distributeddata_diff(
                link_fields,
                graph,
                [&](u64 id) {
                    link_fields.add_obj(
                        id, modules::NeighGraphLinkField<T>{graph.get(id).get(), nvar});
                },
                [&](u64 id) {
                    link_fields.get(id).resize(graph.get(id).get());
                },
                [&](u64 id) {
                    link_fields.erase(id);
                });
        }

        inline void check_size(const shamrock::solvergraph::Indexes<u32> &link_counts) const {

            on_distributeddata_diff(
                link_fields,
                link_counts.indexes,
                [&](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing link field in distributed data at id " + std::to_string(id));
                },
                [&](u64 id) {
                    if (link_fields.get(id).link_count != link_counts.indexes.get(id)) {
                        shambase::throw_with_loc<std::runtime_error>(
                            "Wrong size for link field in distributed data at id "
                            + std::to_string(id));
                    }
                },
                [&](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra link field in distributed data at id " + std::to_string(id));
                });
        }

        inline void resize_according_to(const shamrock::solvergraph::Indexes<u32> &link_counts) {

            on_distributeddata_diff(
                link_fields,
                link_counts.indexes,
                [&](u64 id) {
                    link_fields.add_obj(
                        id, modules::NeighGraphLinkField<T>{link_counts.indexes.get(id), nvar});
                },
                [&](u64 id) {
                    link_fields.get(id).resize(link_counts.indexes.get(id));
                },
                [&](u64 id) {
                    link_fields.erase(id);
                });
        }

        inline virtual void free_alloc() { link_fields = {}; }
    };

} // namespace shammodels::basegodunov::solvergraph
