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
 * @file GhostZoneData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/DistributedDataShared.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/sycl.hpp"
#include "shammath/AABB.hpp"

namespace shammodels::basegodunov {

    /**
     * @brief Class to hold information related to ghost zones
     *
     * @tparam Tvec
     * @tparam TgridVec
     */
    template<class Tvec, class TgridVec>
    class GhostZonesData {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        struct InterfaceBuildInfos {
            TgridVec offset;
            sycl::vec<i32, dim> periodicity_index;
            shammath::AABB<TgridVec> volume_target;
        };

        struct InterfaceIdTable {
            InterfaceBuildInfos build_infos;
            std::unique_ptr<sycl::buffer<u32>> ids_interf;
            f64 cell_count_ratio;
        };

        using GeneratorMap = shambase::DistributedDataShared<InterfaceBuildInfos>;
        GeneratorMap ghost_gen_infos;

        shambase::DistributedDataShared<InterfaceIdTable> ghost_id_build_map;

        template<class T>
        shambase::DistributedDataShared<T> build_interface_native(
            std::function<T(u64, u64, InterfaceBuildInfos, sycl::buffer<u32> &, u32)> fct) {
            StackEntry stack_loc{};

            // clang-format off
            return ghost_id_build_map.template map<T>([&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
                if (!bool(build_table.ids_interf)) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "their is an empty id table in the interface, it should have been removed");
                }

                return fct(
                    sender,
                    receiver,
                    build_table.build_infos,
                    *build_table.ids_interf,
                    build_table.ids_interf->size());

            });
            // clang-format on
        }
    };

} // namespace shammodels::basegodunov
