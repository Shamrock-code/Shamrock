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
 * @file FaceFlagger.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/zeus/NeighFaceList.hpp"
#include "shammodels/zeus/Solver.hpp"
#include "shammodels/zeus/modules/SolverStorage.hpp"

namespace shammodels::zeus::modules {

    /**
     * @brief flag faces with a lookup index for the orientation
     *
     * 0 = x-
     * 1 = x+
     * 2 = y-
     * 3 = y+
     * 4 = z-
     * 5 = z+
     * @tparam Tvec
     * @tparam TgridVec
     */
    template<class Tvec, class TgridVec>
    class FaceFlagger {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        FaceFlagger(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief flag faces with a lookup index
         * performs at around 2G cell per seconds on a RTX A5000
         */
        void flag_faces();

        static constexpr u8 max_lookup = 5;

        inline static Tvec lookup_to_normal(u8 lookup) {
            return std::array<Tvec, 6>{
                Tvec{-1, 0, 0},
                Tvec{1, 0, 0},
                Tvec{0, -1, 0},
                Tvec{0, 1, 0},
                Tvec{0, 0, -1},
                Tvec{0, 0, 1},
            }[lookup];
        }

        using FaceList = NeighFaceList<Tvec>;
        void split_face_list();

        void compute_neigh_ids();

        private:
        shamrock::tree::ObjectCache isolate_lookups(
            shamrock::tree::ObjectCache &cache,
            sycl::buffer<u8> &face_normals_lookup,
            u8 lookup_value);

        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::zeus::modules
