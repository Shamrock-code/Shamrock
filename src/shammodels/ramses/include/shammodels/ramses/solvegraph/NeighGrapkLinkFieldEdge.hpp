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
 * @file NeighGrapkLinkFieldEdge.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/common/amr/NeighGraphLinkField.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamtree/RadixTree.hpp"
#include <functional>

namespace shammodels::basegodunov::solvergraph {

    template<class T>
    class NeighGrapkLinkFieldEdge : public shamrock::solvergraph::IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;

        shambase::DistributedData<modules::NeighGraphLinkField<T>> link_fields;

        inline void resize_according_to(
            const shambase::DistributedData<std::reference_wrapper<modules::AMRGraph>> &graph) {

            on_distributeddata_diff(
                link_fields,
                graph,
                [&](u64 id) {
                    link_fields.add_obj(id, modules::NeighGraphLinkField<T>{graph.get(id).get()});
                },
                [&](u64 id) {
                    link_fields.get(id).resize(graph.get(id).get());
                },
                [&](u64 id) {
                    link_fields.erase(id);
                });
        }

        inline virtual void free_alloc() { link_fields = {}; }
    };

} // namespace shammodels::basegodunov::solvergraph
