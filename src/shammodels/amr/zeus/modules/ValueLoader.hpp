// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/SolverStorage.hpp"

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

        shamrock::ComputeField<T> load_value(
            std::string field_name, std::array<Tgridscal, dim> offset, std::string result_name);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        void load_patch_internal_block(
            std::array<Tgridscal, dim> offset,
            u32 nobj,
            u32 nvar,
            sycl::buffer<T> &src,
            sycl::buffer<T> &dest);

        void load_patch_neigh_same_level(
            std::array<Tgridscal, dim> offset,
            sycl::buffer<TgridVec> &buf_cell_min,
            sycl::buffer<TgridVec> &buf_cell_max,
            shammodels::zeus::NeighFaceList<Tvec> & face_lists,
            u32 nobj,
            u32 nvar,
            sycl::buffer<T> &src,
            sycl::buffer<T> &dest
        );
    };

} // namespace shammodels::zeus::modules