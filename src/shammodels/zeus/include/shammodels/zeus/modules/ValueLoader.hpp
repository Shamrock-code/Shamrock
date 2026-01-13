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
 * @file ValueLoader.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/zeus/Solver.hpp"
#include "shammodels/zeus/modules/SolverStorage.hpp"

namespace shammodels::zeus::modules {

    template<class Tvec, class TgridVec, class T>
    class ValueLoader {

        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        ValueLoader(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief
         * @todo specify multiple function if ghost or no ghost, if source is compute field or
         * something else
         * @param field_name
         * @param offset
         * @param result_name
         * @return shamrock::ComputeField<T>
         */
        shamrock::ComputeField<T> load_value_with_gz(
            std::string field_name, std::array<Tgridscal, dim> offset, std::string result_name);

        shamrock::ComputeField<T> load_value_with_gz(
            shamrock::ComputeField<T> &compute_field,
            std::array<Tgridscal, dim> offset,
            std::string result_name);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        /**
         * @brief load value in its own block
         *
         * @param offset
         * @param nobj
         * @param nvar
         * @param src
         * @param dest
         */
        void load_patch_internal_block(
            std::array<Tgridscal, dim> offset,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        /**
         * @brief load value of neighbour blocks having same level
         *
         * @param offset
         * @param buf_cell_min
         * @param buf_cell_max
         * @param face_lists
         * @param nobj
         * @param nvar
         * @param src
         * @param dest
         */
        void load_patch_neigh_same_level(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        /**
         * @brief load value of neighbour block with level = +1
         *
         * @param offset
         * @param buf_cell_min
         * @param buf_cell_max
         * @param face_lists
         * @param nobj
         * @param nvar
         * @param src
         * @param dest
         */
        void load_patch_neigh_level_up(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        /**
         * @brief load value of neighbour block with level = -1
         *
         * @param offset
         * @param buf_cell_min
         * @param buf_cell_max
         * @param face_lists
         * @param nobj
         * @param nvar
         * @param src
         * @param dest
         */
        void load_patch_neigh_level_down(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        void load_patch_internal_block_xm(
            u32 nobj, u32 nvar, sham::DeviceBuffer<T> &src, sham::DeviceBuffer<T> &dest);

        void load_patch_internal_block_ym(
            u32 nobj, u32 nvar, sham::DeviceBuffer<T> &src, sham::DeviceBuffer<T> &dest);

        void load_patch_internal_block_zm(
            u32 nobj, u32 nvar, sham::DeviceBuffer<T> &src, sham::DeviceBuffer<T> &dest);

        void load_patch_internal_block_xp(
            u32 nobj, u32 nvar, sham::DeviceBuffer<T> &src, sham::DeviceBuffer<T> &dest);

        void load_patch_internal_block_yp(
            u32 nobj, u32 nvar, sham::DeviceBuffer<T> &src, sham::DeviceBuffer<T> &dest);

        void load_patch_internal_block_zp(
            u32 nobj, u32 nvar, sham::DeviceBuffer<T> &src, sham::DeviceBuffer<T> &dest);

        void load_patch_neigh_same_level_xm(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        void load_patch_neigh_same_level_ym(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        void load_patch_neigh_same_level_zm(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        void load_patch_neigh_same_level_xp(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        void load_patch_neigh_same_level_yp(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);

        void load_patch_neigh_same_level_zp(
            std::array<Tgridscal, dim> offset,
            sham::DeviceBuffer<TgridVec> &buf_cell_min,
            sham::DeviceBuffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> &face_lists,
            u32 nobj,
            u32 nvar,
            sham::DeviceBuffer<T> &src,
            sham::DeviceBuffer<T> &dest);
    };

} // namespace shammodels::zeus::modules
