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
 * @file SPHUtilities.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/BuildGhostInterfaceIdTable.hpp"
#include "shammodels/sph/modules/ComputePatchTreeMaxField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/PatchtreeFieldEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <utility>

namespace shammodels::sph {

    /**
     * @brief handle basic utilities dealing with SPH
     *
     * @tparam vec
     */
    template<class vec, class SPHKernel>
    class SPHUtilities {
        public:
        using flt = shambase::VecComponent<vec>;

        static constexpr flt Rkern = SPHKernel::Rkern;

        using GhostHndl = BasicSPHGhostHandler<vec>;
        using InterfBuildCache
            = shambase::DistributedDataShared<typename GhostHndl::InterfaceIdTable>;

        PatchScheduler &sched;

        SPHUtilities(PatchScheduler &sched) : sched(sched) {}

        inline InterfBuildCache build_interf_cache(
            GhostHndl &interf_handle, SerialPatchTree<vec> &sptree, flt h_evol_max) {

            using namespace shamrock::patch;
            using namespace shamrock::solvergraph;

            using InterfaceBuildInfos = typename GhostHndl::InterfaceBuildInfos;
            using InterfaceIdTable    = typename GhostHndl::InterfaceIdTable;

            const u32 ihpart = sched.pdl_old().template get_field_idx<flt>("hpart");

            PatchField<flt> interactR_patch = sched.map_owned_to_patch_field_simple<flt>(
                [&](const Patch p, PatchDataLayer &pdat) -> flt {
                    if (!pdat.is_empty()) {
                        return pdat.get_field<flt>(ihpart).compute_max() * h_evol_max * Rkern;
                    } else {
                        return shambase::VectorProperties<flt>::get_min();
                    }
                });

            // ------------------------------------------------------------------------------------
            // temporary wrapper to slowly migrate to the new solvergraph
            auto patch_tree        = std::make_shared<SerialPatchTreeRefEdge<vec>>("", "");
            patch_tree->patch_tree = std::ref(sptree);

            // borrow the per patch field (O(1) move instead of a copy over all global patches), it
            // is moved back once the node has been evaluated
            auto interact_radius    = std::make_shared<ScalarsEdge<flt>>("", "");
            interact_radius->values = std::move(interactR_patch.field_all);

            auto interact_radius_tree = std::make_shared<PatchtreeFieldEdge<flt>>("", "");

            modules::ComputePatchTreeMaxField<vec> compute_interact_radius_tree;
            compute_interact_radius_tree.set_edges(
                patch_tree, interact_radius, interact_radius_tree);
            compute_interact_radius_tree.evaluate();

            interactR_patch.field_all = std::move(interact_radius->values);

            auto interface_infos    = std::make_shared<DDSharedScalar<InterfaceBuildInfos>>("", "");
            interface_infos->values = interf_handle.find_interfaces(
                sptree, std::move(interact_radius_tree->patchtree_field), interactR_patch);

            auto positions                          = std::make_shared<FieldRefs<vec>>("", "");
            DDPatchDataFieldRef<vec> positions_refs = {};
            sched.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                positions_refs.add_obj(p.id_patch, std::ref(pdat.get_field<vec>(0)));
            });
            positions->set_refs(positions_refs);

            auto interface_id_table = std::make_shared<DDSharedScalar<InterfaceIdTable>>("", "");

            modules::BuildGhostInterfaceIdTable<vec> build_id_table;
            build_id_table.set_edges(positions, interface_infos, interface_id_table);
            build_id_table.evaluate();
            // ------------------------------------------------------------------------------------

            return std::move(interface_id_table->values);
        }

        static void iterate_smoothing_length_cache(

            sham::DeviceBuffer<vec> &merged_r,
            sham::DeviceBuffer<flt> &hnew,
            sham::DeviceBuffer<flt> &hold,
            sham::DeviceBuffer<flt> &eps_h,
            sycl::range<1> update_range,
            shamrock::tree::ObjectCache &neigh_cache,

            flt gpart_mass,
            flt h_evol_max,
            flt h_evol_iter_max

        );

        static void compute_omega(
            sham::DeviceBuffer<vec> &merged_r,
            sham::DeviceBuffer<flt> &h_part,
            sham::DeviceBuffer<flt> &omega_h,
            sycl::range<1> part_range,
            shamrock::tree::ObjectCache &neigh_cache,
            flt gpart_mass);
    };

} // namespace shammodels::sph
