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
 * @file GhostZones.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"

namespace shammodels::basegodunov::modules {

    /**
     * @brief Module of the Godunov solver to deal with ghost zone exchanges
     *
     * This class is responsible for generating and exchanging ghost zones
     * between neighboring patches.
     *
     * @tparam Tvec type of the vector used for the coordinates
     * @tparam TgridVec type of the vector used for the coordinates of the
     * grid
     */
    template<class Tvec, class TgridVec>
    class GhostZones {
        public:
        /// Type for the physical scalars
        using Tscal = shambase::VecComponent<Tvec>;
        /// Type for the AMR grid coordinate scalars
        using Tgridscal = shambase::VecComponent<TgridVec>;
        /// Dimension of the coordinates space
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        /// Alias to the SolverConfig type
        using Config = SolverConfig<Tvec, TgridVec>;
        /// Alias to the SolverStorage type
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        /// Reference to the Shamrock context
        ShamrockCtx &context;
        /// Reference to the configuration of the solver
        Config &solver_config;
        /// Reference to the storage of the solver
        Storage &storage;

        /**
         * @brief Constructor of the module
         *
         * @param context reference to the context of Shamrock
         * @param solver_config reference to the configuration of the solver
         * @param storage reference to the storage of the solver
         */
        GhostZones(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief Build the cache of ghost zones
         *
         * The cache of ghost zones is built by calling the function
         * `find_interfaces` that returns a map of patch id to a vector of
         * `InterfaceBuildInfos`, which contains the information to build
         * the interface data.
         *
         * The function loops over the map and for each interface calls the
         * function `build_interface_native` that builds the interface objects ids
         * and stores it in the cache.
         */
        void build_ghost_cache();

        /**
         * @brief Communicate patch datas (having layout pdl) according to the object ids compute
         * using `build_ghost_cache()`
         *
         * This function is typically used to communicate patch data using the
         * ghost_patch_data_layout.
         *
         * @param pdl the patch data layout
         * @param interf the set of patch data to communicate and merge
         * @return The patch data of the interfaces
         */
        shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> communicate_pdat(
            shamrock::patch::PatchDataLayerLayout &pdl,
            shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> &&interf);

        /**
         * @brief Communicate a single patch data field according to the object ids compute using
         * `build_ghost_cache()`
         *
         * This function is similar to `communicate_pdat` except that it communicates only one field
         * per patches instead of the complete patch data.
         *
         * @param interf the patch data field to communicate and merge
         * @return the communicates patch data fields
         */
        template<class T>
        shambase::DistributedDataShared<PatchDataField<T>>
        communicate_pdat_field(shambase::DistributedDataShared<PatchDataField<T>> &&interf);

        /**
         * @brief Exchange the ghost zones of a given compute field and return the merged data after
         * the exchange
         *
         * @param in compute field to exchange
         * @return the exchanged compute field
         */
        template<class T>
        shamrock::ComputeField<T> exchange_compute_field(shamrock::ComputeField<T> &in);

        /**
         * @brief Merge intefaces into their corresponding patch, and return the merged patches
         *
         * @param interfs set of data of the intefaces
         * @param init function to initialize the merged patch data field
         * @param appender function to append the data to the merged object
         * @return the object with merged interfaces
         */
        template<class T, class Tmerged>
        shambase::DistributedData<Tmerged> merge_native(
            shambase::DistributedDataShared<T> &&interfs,
            std::function<
                Tmerged(const shamrock::patch::Patch, shamrock::patch::PatchDataLayer &pdat)> init,
            std::function<void(Tmerged &, T &)> appender);

        /// @brief Exchange the ghost zones of the solver
        void exchange_ghost();

        private:
        /// Get a reference to the scheduler of Shamrock
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::basegodunov::modules
