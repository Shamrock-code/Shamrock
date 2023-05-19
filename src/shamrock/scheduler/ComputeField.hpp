// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambase/sycl.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamrock/math/integrators.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock {

    template<class T>
    class ComputeField {

        public:
        shambase::DistributedData<PatchDataField<T>> field_data;

        inline void generate(PatchScheduler &sched, std::string name) {StackEntry stack_loc{};

            using namespace shamrock::patch;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                field_data.insert({id_patch, PatchDataField<T>(name, 1)});
                field_data.get(id_patch).resize(pdat.get_obj_cnt());
            });
        }

        inline const std::unique_ptr<sycl::buffer<T>> &get_buf(u64 id_patch) {
            return field_data.get(id_patch).get_buf();
        }

        inline PatchDataField<T> &get_field(u64 id_patch) { return field_data.get(id_patch); }

        inline T compute_rank_max(){StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_min();
            field_data.for_each([&](u64 id, PatchDataField<T> & cfield){
                if(!cfield.is_empty()){
                    ret = shambase::sycl_utils::g_sycl_max(ret, cfield.compute_max());
                }
            });

            return ret;
        }

        inline T compute_rank_min(){StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_max();
            field_data.for_each([&](u64 id, PatchDataField<T> & cfield){
                if(!cfield.is_empty()){
                    ret = shambase::sycl_utils::g_sycl_min(ret, cfield.compute_min());
                }
            });

            return ret;
        }

        inline T compute_rank_sum(){StackEntry stack_loc{};
            T ret = shambase::VectorProperties<T>::get_zero();
            field_data.for_each([&](u64 id, PatchDataField<T> & cfield){
                if(!cfield.is_empty()){
                    ret = shambase::sycl_utils::g_sycl_min(ret, cfield.compute_min());
                }
            });

            return ret;
        }

        inline void reset(){
            field_data.reset();
        }
    };
}