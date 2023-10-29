// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GhostZoneData.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/DistributedData.hpp"
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
    };

} // namespace shammodels::basegodunov